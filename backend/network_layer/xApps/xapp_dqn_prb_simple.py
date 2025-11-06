"""Simplified DQN-based PRB allocator.

This variant keeps the action/state definitions of the original xApp but strips
away auxiliary heads, sequence encoders, horizon buffering, and telemetry so it
can serve as a clean baseline for experimentation.
"""

from .xapp_base import xAppBase

import math
import os
import random
from collections import deque, defaultdict
from itertools import product
from typing import Dict, Optional

import settings
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("xapp_dqn_prb_simple: torch not available; xApp will disable itself.")


SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")
MAX_UE_NORM = 50.0


class _ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self._buf = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch: int):
        batch = min(batch, len(self._buf))
        idx = np.random.choice(len(self._buf), batch, replace=False)
        s, a, r, ns, d = zip(*[self._buf[i] for i in idx])
        s_t = torch.tensor(np.stack(s), dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.long)
        r_t = torch.tensor(r, dtype=torch.float32)
        ns_t = torch.tensor(np.stack(ns), dtype=torch.float32)
        d_t = torch.tensor(d, dtype=torch.float32)
        return s_t, a_t, r_t, ns_t, d_t

    def __len__(self):
        return len(self._buf)


class _SimpleQNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class xAppSimpleDQNPRBAllocator(xAppBase):
    """Minimal DQN PRB allocator (MLP + replay buffer)."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        base_enabled = getattr(settings, "DQN_PRB_ENABLE", False)
        simple_enabled = getattr(settings, "DQN_PRB_SIMPLE_ENABLE", False)
        self.enabled = base_enabled and simple_enabled
        if not self.enabled:
            return
        if not TORCH_AVAILABLE:
            self.enabled = False
            print(f"{self.xapp_id}: torch unavailable; disabling simplified DQN xApp.")
            return

        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-3))
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50000))
        self.target_sync_interval = max(1, int(getattr(settings, "DQN_PRB_TARGET_UPDATE", 200)))
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10000))
        self.save_interval = int(getattr(settings, "DQN_PRB_SAVE_INTERVAL", 0))
        if self.save_interval < 0:
            self.save_interval = 0
        self.train_max_steps = max(0, int(getattr(settings, "DQN_PRB_MAX_TRAIN_STEPS", 0)))

        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))
        self.need_saturation = max(1e-6, float(getattr(settings, "DQN_NEED_SATURATION", 1.5)))
        self.norm_dl_mbps = float(getattr(settings, "DQN_NORM_MAX_DL_MBPS", 100.0))
        self.norm_buf_bytes = float(getattr(settings, "DQN_NORM_MAX_BUF_BYTES", 1e6))

        self._t = 0
        self._last_loss: Optional[float] = None
        self._last_reward: Optional[float] = None
        self._action_counts = defaultdict(int)
        self._prev_decision: Dict[str, Dict[str, np.ndarray]] = {}
        self._device = self._select_device()
        self._state_dim = 18
        self._action_combos = list(product([-1, 0, 1], repeat=3))
        self._action_labels = [
            f"Δe={combo[0]},Δu={combo[1]},Δm={combo[2]}" for combo in self._action_combos
        ]
        self._n_actions = len(self._action_combos)
        self._buffer = _ReplayBuffer(self.buffer_cap)

        self._policy = _SimpleQNetwork(self._state_dim, self._n_actions).to(self._device)
        self._target = _SimpleQNetwork(self._state_dim, self._n_actions).to(self._device)
        self._target.load_state_dict(self._policy.state_dict())
        self._opt = optim.Adam(self._policy.parameters(), lr=self.lr)

        base_model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")
        root, ext = os.path.splitext(base_model_path)
        ext = ext or ".pt"
        if not root.endswith("_simple"):
            root = f"{root}_simple"
        self.model_path = f"{root}{ext}"
        if self.train_mode:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self._maybe_load_weights()

    # ---------------- xApp lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        print(f"{self.xapp_id}: enabled (train={self.train_mode}, period={self.period_steps} steps)")

    # ---------------- Core helpers ----------------
    def _select_device(self):
        pref = getattr(settings, "DQN_PRB_DEVICE", "auto") or "auto"
        pref = str(pref).strip().lower()
        if pref in {"auto", "", "default"}:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        try:
            device = torch.device(str(pref))
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return device
        except Exception as exc:
            print(f"{self.xapp_id}: requested device '{pref}' unavailable ({exc}); using CPU.")
            return torch.device("cpu")

    def _epsilon(self):
        if self.eps_decay <= 0:
            return self.eps_end
        frac = min(1.0, self._t / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _act(self, state_vec: np.ndarray) -> int:
        if not TORCH_AVAILABLE:
            return random.randrange(self._n_actions)
        eps = self._epsilon() if self.train_mode else 0.0
        if random.random() < eps:
            return random.randrange(self._n_actions)
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._policy(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _train_step(self):
        if not self.train_mode or len(self._buffer) < max(32, self.batch):
            return None
        s, a, r, ns, d = self._buffer.sample(self.batch)
        s = s.to(self._device)
        a = a.to(self._device)
        r = r.to(self._device)
        ns = ns.to(self._device)
        d = d.to(self._device)

        q_pred = self._policy(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self._target(ns).max(1)[0]
            target = r + (1.0 - d) * self.gamma * next_q
        loss = F.mse_loss(q_pred, target)
        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._opt.step()

        if self._t % self.target_sync_interval == 0:
            self._target.load_state_dict(self._policy.state_dict())

        self._last_loss = float(loss.detach().item())
        return self._last_loss

    def _maybe_load_weights(self):
        if not os.path.exists(self.model_path):
            return
        try:
            state_dict = torch.load(self.model_path, map_location=self._device)
            self._policy.load_state_dict(state_dict)
            self._target.load_state_dict(self._policy.state_dict())
            print(f"{self.xapp_id}: loaded weights from {self.model_path}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to load weights ({exc})")

    def _save_model(self, path: str):
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            torch.save(self._policy.state_dict(), path)
        except Exception as exc:
            print(f"{self.xapp_id}: failed to save model ({exc})")

    def _maybe_save(self):
        if not self.train_mode or self.save_interval <= 0:
            return
        if self._t <= 0:
            return
        if (self._t % self.save_interval) != 0:
            return
        self._save_model(self.model_path)

    # ---------------- Simulation hooks ----------------
    def step(self):
        if not self.enabled:
            return
        sim_engine = getattr(self.ric, "simulation_engine", None)
        sim_step = getattr(sim_engine, "sim_step", 0)
        decision_due = (self.period_steps <= 1) or (sim_step % self.period_steps == 0)
        if not decision_due:
            return
        if self.train_mode and self.train_max_steps > 0 and self._t >= self.train_max_steps:
            self.enabled = False
            print(f"{self.xapp_id}: reached max train steps ({self.train_max_steps}); stopping updates.")
            return

        self._t += 1
        step_dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1))
        period_dt = step_dt * float(self.period_steps)

        for cell_id, cell in self.cell_list.items():
            state_vec = np.asarray(self._get_state(cell), dtype=np.float32)
            prev = self._prev_decision.get(cell_id)
            if prev is not None:
                reward = self._reward(cell, period_dt)
                if self.train_mode:
                    self._buffer.push(prev["state"], prev["action"], reward, state_vec, 0.0)
                self._last_reward = reward

            if self.train_mode:
                self._train_step()

            action = self._act(state_vec)
            self._apply_action(cell, action)
            self._action_counts[action] += 1
            self._prev_decision[cell_id] = {"state": state_vec.copy(), "action": action}

        self._maybe_save()

    # ---------------- State, reward, and actions ----------------
    def _get_slice_counts(self, cell):
        cnt = {SL_E: 0, SL_U: 0, SL_M: 0}
        for ue in getattr(cell, "connected_ue_list", {}).values():
            sl = getattr(ue, "slice_type", None)
            if sl in cnt:
                cnt[sl] += 1
        return cnt

    def _get_state(self, cell):
        cnt = self._get_slice_counts(cell)
        agg = self._aggregate_slice_metrics(cell)
        max_ue = MAX_UE_NORM
        max_prb = max(1.0, float(getattr(cell, "max_dl_prb", 1)))

        def clamp01(val: float) -> float:
            return max(0.0, min(1.0, float(val)))

        slice_order = (SL_M, SL_U, SL_E)
        state = []

        for sl in slice_order:
            val = clamp01(float(cnt.get(sl, 0.0)) / max_ue)
            state.append(val)

        for sl in slice_order:
            val = clamp01(float(agg.get(sl, {}).get("slice_prb", 0.0)) / max_prb)
            state.append(val)

        dl_norm = max(1e-9, self.norm_dl_mbps)
        for sl in slice_order:
            val = clamp01(float(agg.get(sl, {}).get("tx_mbps", 0.0)) / dl_norm)
            state.append(val)

        buf_norm = max(1e-9, self.norm_buf_bytes)
        for sl in slice_order:
            val = clamp01(float(agg.get(sl, {}).get("buf_bytes", 0.0)) / buf_norm)
            state.append(val)

        for sl in slice_order:
            val = clamp01(float(agg.get(sl, {}).get("prb_req", 0.0)) / max_prb)
            state.append(val)

        for sl in slice_order:
            req = float(agg.get(sl, {}).get("prb_req", 0.0))
            granted = float(agg.get(sl, {}).get("prb_granted", 0.0))
            if req <= 0.0:
                ratio = 1.0 if granted <= 0.0 else granted / max_prb
            else:
                ratio = granted / max(1.0, req)
            state.append(clamp01(ratio))

        return state

    def _aggregate_slice_metrics(self, cell):
        agg = {
            sl: {
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "prb_req": 0.0,
                "prb_granted": 0.0,
                "slice_prb": 0.0,
                "latency_sum": 0.0,
                "latency_count": 0,
            }
            for sl in (SL_E, SL_U, SL_M)
        }

        quota_map = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        max_prb = float(getattr(cell, "max_dl_prb", 0) or 0.0)
        if SL_E not in quota_map:
            other = float(quota_map.get(SL_U, 0.0)) + float(quota_map.get(SL_M, 0.0))
            quota_map[SL_E] = max(0.0, max_prb - other)
        for sl in (SL_E, SL_U, SL_M):
            agg[sl]["slice_prb"] = float(quota_map.get(sl, 0.0))

        req_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        alloc_map = getattr(cell, "prb_ue_allocation_dict", {}) or {}

        for ue in getattr(cell, "connected_ue_list", {}).values():
            sl = getattr(ue, "slice_type", None)
            if sl not in agg:
                continue
            agg[sl]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
            agg[sl]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            imsi = getattr(ue, "ue_imsi", None)
            if imsi is None:
                continue
            agg[sl]["prb_req"] += float(req_map.get(imsi, 0.0) or 0.0)
            alloc = alloc_map.get(imsi, {}) or {}
            agg[sl]["prb_granted"] += float(alloc.get("downlink", 0.0) or 0.0)
            agg[sl]["latency_sum"] += float(getattr(ue, "downlink_latency", 0.0) or 0.0)
            agg[sl]["latency_count"] += 1

        return agg

    def _slice_scores(self, cell, T_s):
        agg = self._aggregate_slice_metrics(cell)

        def clamp01(val: float) -> float:
            return max(0.0, min(1.0, float(val)))

        scores = {}
        for sl in (SL_E, SL_U, SL_M):
            data = agg[sl]
            quota_raw = float(max(0.0, data["slice_prb"]))
            quota_safe = max(1.0, quota_raw)
            demand = max(0.0, float(data["prb_req"]))
            granted = max(0.0, float(data["prb_granted"]))
            buf_bytes = max(0.0, float(data["buf_bytes"]))
            tx_mbps = max(0.0, float(data["tx_mbps"]))

            throughput_norm = clamp01(tx_mbps / max(1e-9, self.norm_dl_mbps))
            backlog_norm = clamp01(buf_bytes / max(1e-9, self.norm_buf_bytes))
            if demand <= 0.0:
                satisfaction = 1.0 if granted <= 0.0 else clamp01(granted / quota_safe)
            else:
                satisfaction = clamp01(granted / max(1.0, demand))

            if demand <= 0.0:
                need_ratio = 0.0
            elif quota_raw <= 0.0:
                need_ratio = self.need_saturation
            else:
                need_ratio = demand / max(1e-6, quota_raw)
            need = clamp01(need_ratio / max(1e-6, self.need_saturation))

            oversupply = clamp01(max(0.0, granted - demand) / quota_safe)
            idle = clamp01(max(0.0, quota_raw - granted) / quota_safe)

            latency_avg = 0.0
            lat_count = int(data.get("latency_count", 0) or 0)
            if lat_count > 0:
                latency_avg = float(data.get("latency_sum", 0.0) or 0.0) / max(1, lat_count)

            scores[sl] = {
                "throughput": throughput_norm,
                "backlog": backlog_norm,
                "satisfaction": satisfaction,
                "need": need,
                "oversupply": oversupply,
                "idle": idle,
                "buf_bytes": buf_bytes,
                "tx_mbps": tx_mbps,
                "latency": latency_avg,
            }

        embb = scores[SL_E]
        embb_score = (
            0.55 * embb["satisfaction"] + 0.25 * embb["throughput"] + 0.20 * (1.0 - embb["backlog"])
        )
        embb_score -= 0.10 * embb["oversupply"]
        embb_score = clamp01(embb_score)

        urllc = scores[SL_U]
        buf_bits = urllc["buf_bytes"] * 8.0
        tx_bps = urllc["tx_mbps"] * 1e6
        if tx_bps <= 0.0:
            delay_s = float("inf") if buf_bits > 0.0 else 0.0
        else:
            delay_s = buf_bits / max(1e-9, tx_bps)
        gamma = max(1e-12, self.urlc_gamma_s)
        delay_term = math.exp(-delay_s / gamma) if math.isfinite(delay_s) else 0.0
        delay_term = clamp01(delay_term)
        urllc_score = (
            0.60 * delay_term + 0.30 * urllc["satisfaction"] + 0.10 * (1.0 - urllc["backlog"])
        )
        urllc_score -= 0.05 * urllc["oversupply"]
        urllc_score = clamp01(urllc_score)

        mmtc = scores[SL_M]
        mmtc_score = (
            0.50 * mmtc["satisfaction"]
            + 0.30 * (1.0 - mmtc["idle"])
            + 0.20 * (1.0 - mmtc["backlog"])
        )
        mmtc_score -= 0.10 * mmtc["oversupply"]
        mmtc_score = clamp01(mmtc_score)

        return embb_score, urllc_score, mmtc_score

    def _reward(self, cell, T_s):
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _apply_action(self, cell, action_idx: int):
        combo = self._action_combos[int(action_idx)]
        deltas = {SL_E: combo[0], SL_U: combo[1], SL_M: combo[2]}

        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        for sl in (SL_E, SL_U, SL_M):
            quotas.setdefault(sl, 0)
        max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
        if max_prb <= 0:
            return False

        delta_prb = max(1, int(self.move_step))
        initial = {sl: int(quotas.get(sl, 0)) for sl in (SL_E, SL_U, SL_M)}
        total = sum(quotas.values())

        for sl, sign in deltas.items():
            if sign < 0:
                reduce = min(delta_prb * abs(sign), quotas[sl])
                if reduce > 0:
                    quotas[sl] -= reduce
                    total -= reduce

        requested = {sl: delta_prb * sign for sl, sign in deltas.items() if sign > 0}
        desired = {sl: float(quotas[sl]) for sl in (SL_E, SL_U, SL_M)}
        for sl, inc in requested.items():
            desired[sl] = desired.get(sl, 0.0) + float(inc)

        desired_sum = sum(desired.values())
        final = {sl: int(max(0, quotas.get(sl, 0))) for sl in (SL_E, SL_U, SL_M)}
        if requested:
            if desired_sum <= max_prb:
                final = {sl: int(max(0, desired.get(sl, 0.0))) for sl in (SL_E, SL_U, SL_M)}
            elif desired_sum > 0:
                scale = max_prb / float(desired_sum)
                fractional = []
                assigned = 0
                scaled = {}
                for sl in (SL_E, SL_U, SL_M):
                    raw = desired.get(sl, 0.0) * scale
                    tgt = int(math.floor(raw))
                    scaled[sl] = tgt
                    assigned += tgt
                    fractional.append((sl, raw - tgt))
                remainder = max(0, max_prb - assigned)
                if remainder > 0:
                    fractional.sort(key=lambda item: item[1], reverse=True)
                    for sl, _frac in fractional:
                        if remainder <= 0:
                            break
                        scaled[sl] += 1
                        remainder -= 1
                final = {sl: int(max(0, scaled.get(sl, 0))) for sl in (SL_E, SL_U, SL_M)}

        changed = any(final[sl] != initial.get(sl, 0) for sl in (SL_E, SL_U, SL_M))
        if not changed:
            try:
                setattr(
                    cell,
                    "rl_last_action",
                    {
                        "actor": "DQN_SIMPLE",
                        "code": int(action_idx),
                        "label": self._action_labels[int(action_idx)],
                        "changed": False,
                        "quota": dict(getattr(cell, "slice_dl_prb_quota", {}) or {}),
                    },
                )
            except Exception:
                pass
            return False

        cell.slice_dl_prb_quota = {s: int(max(0, final.get(s, 0))) for s in final}
        try:
            setattr(
                cell,
                "rl_last_action",
                {
                    "actor": "DQN_SIMPLE",
                    "code": int(action_idx),
                    "label": self._action_labels[int(action_idx)],
                    "changed": True,
                    "quota": dict(cell.slice_dl_prb_quota),
                },
            )
        except Exception:
            pass
        return True
