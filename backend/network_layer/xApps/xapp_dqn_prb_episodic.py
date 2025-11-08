import json
from typing import Dict, List, Optional, Set, Tuple

import settings
from utils import load_raw_packet_csv

from . import xapp_dqn_prb_allocator as base_alloc

SL_E = base_alloc.SL_E
SL_U = base_alloc.SL_U
SL_M = getattr(base_alloc, "SL_M", None)


class xAppEpisodicDQNPRBAllocator(base_alloc.xAppDQNPRBAllocator):
    """Episodic wrapper around the DQN PRB allocator.

    Each episode bootstraps a deterministic set of UEs/traces/PRB quotas from a JSON
    specification, runs for a fixed number of decision steps, and then resets the
    simulator state before loading the next scenario.
    """

    def __init__(self, ric=None):
        episodic_flag = getattr(settings, "DQN_PRB_EPISODIC_ENABLE", False)
        super().__init__(ric=ric, force_enable=episodic_flag)
        if not episodic_flag:
            self.enabled = False
            return
        if not self.enabled:
            return

        self._episodes = self._load_episode_specs()
        if not self._episodes:
            print(f"{self.xapp_id}: no episodic configuration provided; disabling.")
            self.enabled = False
            return

        self._episode_loop = getattr(settings, "DQN_EPISODE_LOOP", False)
        self._episode_idx = -1
        self._episode_steps_left = 0
        self._active_episode: Optional[Dict] = None
        self._managed_ues: Set[str] = set()
        self._trace_cache: Dict[Tuple[str, str, float], List[Tuple[float, float, float]]] = {}
        self._mark_done_this_step = False

    def start(self):
        super().start()
        if self.enabled:
            print(f"{self.xapp_id}: loaded {len(self._episodes)} episodic scenarios (loop={self._episode_loop}).")

    def step(self):
        if not self.enabled:
            return
        sim_engine = getattr(self.ric, "simulation_engine", None)
        if sim_engine is None:
            return
        if not self._active_episode:
            if not self._start_next_episode(sim_engine):
                return

        decision_due = self._decision_due(sim_engine)
        final_step = decision_due and self._episode_steps_left == 1
        if final_step:
            self._mark_done_this_step = True

        self._enforce_episode_population(sim_engine)

        prev_t = self._t
        super().step()

        if decision_due and self._t > prev_t:
            self._episode_steps_left -= 1

        if self._mark_done_this_step and self._t > prev_t:
            self._mark_done_this_step = False

        if final_step and self._episode_steps_left == 0 and self._t > prev_t:
            self._finalize_episode(sim_engine)
            self._start_next_episode(sim_engine)

    def _decision_due(self, sim_engine) -> bool:
        sim_step = getattr(sim_engine, "sim_step", 0)
        return (self.period_steps <= 1) or (sim_step % self.period_steps == 0)

    def _load_episode_specs(self) -> List[Dict]:
        raw = None
        inline = getattr(settings, "DQN_EPISODE_CONFIG_JSON", "").strip()
        path = getattr(settings, "DQN_EPISODE_CONFIG_PATH", "").strip()
        if inline:
            try:
                raw = json.loads(inline)
            except Exception as exc:
                print(f"{self.xapp_id}: failed to parse DQN_EPISODE_CONFIG_JSON: {exc}")
                raw = None
        elif path:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
            except FileNotFoundError:
                print(f"{self.xapp_id}: episodic config not found at {path}")
            except Exception as exc:
                print(f"{self.xapp_id}: failed to load episodic config {path}: {exc}")
        if raw is None:
            return []

        if isinstance(raw, dict):
            entries = raw.get("episodes") or []
        elif isinstance(raw, list):
            entries = raw
        else:
            print(f"{self.xapp_id}: episodic config must be a list or dict with 'episodes'.")
            return []

        specs = []
        for idx, entry in enumerate(entries):
            spec = self._normalize_episode_entry(entry, idx)
            if spec is not None:
                specs.append(spec)
        return specs

    def _normalize_episode_entry(self, entry, idx: int) -> Optional[Dict]:
        if not isinstance(entry, dict):
            return None
        groups = entry.get("ue_groups") or []
        if not groups:
            print(f"{self.xapp_id}: episode {idx} skipped (ue_groups empty).")
            return None

        norm_groups = []
        for g_idx, group in enumerate(groups):
            sl_name = self._normalise_slice_name(group.get("slice"))
            if sl_name is None:
                print(f"{self.xapp_id}: episode {idx} group {g_idx} has unknown slice; skipping group.")
                continue
            count = int(group.get("count", 1))
            if count <= 0:
                continue
            norm_groups.append({
                "slice": sl_name,
                "count": count,
                "trace_file": group.get("trace") or group.get("trace_file"),
                "ue_ip": group.get("ue_ip"),
                "trace_bin": float(group.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0))),
                "trace_speedup": float(group.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0))),
            })

        if not norm_groups:
            print(f"{self.xapp_id}: episode {idx} skipped (no valid UE groups).")
            return None

        duration = int(entry.get("duration_steps", 0))
        if duration <= 0:
            duration = 1

        slice_prb = {}
        for sl in (SL_E, SL_U, SL_M):
            if not sl:
                continue
            if sl in entry.get("slice_prb", {}):
                slice_prb[sl] = int(entry["slice_prb"][sl])

        return {
            "id": entry.get("id") or f"episode_{idx}",
            "duration_steps": duration,
            "ue_groups": norm_groups,
            "slice_prb": slice_prb,
            "freeze_mobility": bool(entry.get("freeze_mobility", False)),
            "ue_prb_cap": entry.get("ue_prb_cap"),
        }

    def _normalise_slice_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        key = name.strip().lower()
        if key in ("embb", SL_E.lower()):
            return SL_E
        if key in ("urllc", SL_U.lower()):
            return SL_U
        if SL_M and key in ("mmtc", SL_M.lower()):
            return SL_M
        return None

    def _start_next_episode(self, sim_engine) -> bool:
        if not self._episodes:
            return False
        next_idx = self._episode_idx + 1
        if next_idx >= len(self._episodes):
            if not self._episode_loop:
                print(f"{self.xapp_id}: all episodic scenarios consumed; disabling xApp.")
                self.enabled = False
                self._active_episode = None
                return False
            next_idx = 0
        self._episode_idx = next_idx
        self._active_episode = self._episodes[next_idx]
        self._episode_steps_left = self._active_episode["duration_steps"]
        self._deploy_episode(sim_engine, self._active_episode)
        print(f"{self.xapp_id}: started episode '{self._active_episode['id']}' ({self._episode_steps_left} decisions).")
        return True

    def _deploy_episode(self, sim_engine, spec: Dict):
        self._clear_all_ues(sim_engine)
        self._apply_slice_prb(spec, sim_engine)
        self._managed_ues.clear()
        for group in spec["ue_groups"]:
            for idx in range(group["count"]):
                imsi = f"EP_{self._episode_idx}_{group['slice']}_{idx}"
                self._spawn_episode_ue(sim_engine, imsi, group, spec)

    def _clear_all_ues(self, sim_engine):
        for imsi in list(sim_engine.ue_list.keys()):
            sim_engine.deregister_ue(imsi)
        self._managed_ues.clear()

    def _apply_slice_prb(self, spec: Dict, sim_engine):
        targets = spec.get("slice_prb") or {}
        cap = spec.get("ue_prb_cap")
        for cell in sim_engine.cell_list.values():
            if cap is not None:
                cell.prb_per_ue_cap = max(0, int(cap))
            if not targets:
                continue
            updated = dict(cell.slice_dl_prb_quota)
            total = 0
            for sl, val in targets.items():
                prb = max(0, int(val))
                updated[sl] = prb
                total += prb
            max_prb = max(1, int(getattr(cell, "max_dl_prb", 1)))
            if total > max_prb:
                scale = max_prb / float(total)
                for sl in updated:
                    updated[sl] = int(updated[sl] * scale)
            cell.slice_dl_prb_quota = updated

    def _spawn_episode_ue(self, sim_engine, imsi: str, group: Dict, spec: Dict):
        slice_name = group["slice"]
        ok = sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
        if not ok:
            return
        ue = sim_engine.ue_list.get(imsi)
        if ue is None:
            return
        if spec.get("freeze_mobility", False):
            ue.speed_mps = 0
            try:
                ue.set_target(ue.position_x, ue.position_y)
            except Exception:
                pass
        self._managed_ues.add(imsi)
        if group.get("trace_file"):
            self._attach_trace_to_ue(ue, group)

    def _attach_trace_to_ue(self, ue, group: Dict):
        path = group.get("trace_file")
        if not path:
            return
        bin_s = float(group.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)))
        ue_ip = group.get("ue_ip")
        samples = self._load_trace_samples(path, ue_ip, bin_s)
        if not samples:
            print(f"{self.xapp_id}: unable to attach trace {path} to {ue.ue_imsi} (empty samples).")
            return
        speed = float(group.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        bs = ue.current_bs
        try:
            if bs and hasattr(bs, "attach_dl_trace"):
                bs.attach_dl_trace(ue.ue_imsi, samples, speed)
            else:
                ue.attach_trace(samples, speed)
        except Exception as exc:
            print(f"{self.xapp_id}: failed to attach trace {path} to {ue.ue_imsi}: {exc}")

    def _load_trace_samples(self, path: str, ue_ip: Optional[str], bin_s: float):
        key = (path, ue_ip or "", float(bin_s))
        if key not in self._trace_cache:
            try:
                overhead = getattr(settings, "TRACE_OVERHEAD_BYTES", 0)
                samples = load_raw_packet_csv(
                    path,
                    ue_ip=ue_ip,
                    bin_s=float(bin_s),
                    overhead_sub_bytes=int(overhead),
                )
            except Exception as exc:
                print(f"{self.xapp_id}: failed parsing trace {path}: {exc}")
                samples = []
            self._trace_cache[key] = samples
        return self._trace_cache[key]

    def _enforce_episode_population(self, sim_engine):
        extras = [
            imsi for imsi in sim_engine.ue_list.keys()
            if imsi not in self._managed_ues
        ]
        for imsi in extras:
            sim_engine.deregister_ue(imsi)

    def _finalize_episode(self, sim_engine):
        print(f"{self.xapp_id}: completed episode '{self._active_episode['id']}'.")
        self._clear_all_ues(sim_engine)
        self._active_episode = None
        self._episode_steps_left = 0
        self._per_cell_prev.clear()
        self._state_history.clear()
        self._pending_transitions.clear()

    def _queue_transition(self, cell_id, state, action, reward, next_state, done_flag):
        if self._mark_done_this_step:
            done_flag = 1.0
        super()._queue_transition(cell_id, state, action, reward, next_state, done_flag)
