"""Stable-Baselines3 DQN-based PRB allocator xApp.

This module mirrors the behaviour of ``xapp_dqn_prb_allocator`` but delegates the
value function approximation and optimisation loops to Stable-Baselines3's DQN
implementation. The goal is to offer a drop-in alternative that allows
cross-checking policies trained with SB3 against the custom PyTorch variant.

If Stable-Baselines3 (and its Gymnasium dependency) are not available at
runtime the xApp disables itself gracefully.
"""

import os
import random
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional
from itertools import product
import numpy as np

import settings

from .xapp_base import xAppBase

try:
    from gymnasium import Env, spaces
    from stable_baselines3 import DQN as SB3DQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False
    spaces = None  # type: ignore[assignment]

# Reuse slice labels from settings (keep fallbacks identical to the original xApp)
SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")

MAX_UE_NORM = 50.0  # match the custom DQN's fixed UE-count normalization scale


def _linear_eps(start: float, end: float, decay_steps: int, step: int) -> float:
    """Utility: linearly interpolate epsilon for exploration."""
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / float(decay_steps))
    return start + (end - start) * frac


if SB3_AVAILABLE:

    class _StaticPRBEnv(Env):
        """Minimal Gymnasium env used solely to initialise SB3's DQN structures."""

        metadata = {"render_modes": []}

        def __init__(self, obs_dim: int, n_actions: int):
            super().__init__()
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(n_actions)

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
            if seed is not None:
                super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):  # type: ignore[override]
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = False
            info: Dict = {}
            return obs, reward, terminated, truncated, info

else:

    class _StaticPRBEnv:  # type: ignore[too-few-public-methods]
        """Placeholder so type checkers know the symbol exists when SB3 is missing."""

        def __init__(self, *_, **__):
            raise RuntimeError("Stable-Baselines3 is required for xAppSB3DQNPRBAllocator")


class xAppSB3DQNPRBAllocator(xAppBase):
    """PRB allocator powered by Stable-Baselines3 DQN."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)

        self.enabled = getattr(settings, "SB3_DQN_PRB_ENABLE", False)
        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))
        self.norm_dl_mbps = float(getattr(settings, "DQN_NORM_MAX_DL_MBPS", 100.0))
        self.norm_buf_bytes = float(getattr(settings, "DQN_NORM_MAX_BUF_BYTES", 1e6))
        self.need_saturation = max(1e-6, float(getattr(settings, "DQN_NEED_SATURATION", 1.5)))
        self.model_arch = getattr(settings, "DQN_PRB_MODEL_ARCH", "mlp").lower()

        # Reward weights mirror the custom DQN xApp
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))

        # Hyperparameters shared with the PyTorch implementation for easy comparison
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-2))
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50_000))
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10_000))

        default_model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")
        self.model_path = getattr(settings, "SB3_DQN_MODEL_PATH", default_model_path.replace(".pt", "_sb3.zip"))
        self.sb3_total_steps = int(getattr(settings, "SB3_DQN_TOTAL_STEPS", 100_000))
        self.sb3_target_update = int(getattr(settings, "SB3_DQN_TARGET_UPDATE", 1_000))
        self.sb3_log_interval = int(getattr(settings, "SB3_DQN_SAVE_INTERVAL", 5_000))

        self._t = 0
        self._per_cell_prev: Dict[str, Dict] = {}
        self._action_counts = defaultdict(int)

        self._state_dim = 18
        self._action_combos = list(product([-1, 0, 1], repeat=3))
        self._action_labels = [
            f"Δe={combo[0]},Δu={combo[1]},Δm={combo[2]}"
            for combo in self._action_combos
        ]
        self._n_actions = len(self._action_combos)

        self._sb3_env = None
        self._model: Optional[SB3DQN] = None
        self._tb = None
        self._wandb = None
        self._last_loss: Optional[float] = None
        self._last_eps: float = 0.0

        if not self.enabled:
            return

        if not SB3_AVAILABLE:
            print(f"{self.xapp_id}: Stable-Baselines3 not available; disabling xApp.")
            self.enabled = False
            return

        # Optional TensorBoard support (re-using the same flag names as the custom xApp)
        if getattr(settings, "DQN_TB_ENABLE", False):
            try:
                from torch.utils.tensorboard import SummaryWriter

                base = getattr(settings, "DQN_TB_DIR", "backend/tb_logs")
                run = datetime.now().strftime("%Y%m%d_%H%M%S")
                logdir = os.path.join(base, f"sb3_dqn_prb_{run}")
                os.makedirs(logdir, exist_ok=True)
                self._tb = SummaryWriter(log_dir=logdir)
                print(f"{self.xapp_id}: TensorBoard logging to {logdir}")
            except Exception as exc:
                print(f"{self.xapp_id}: TensorBoard unavailable ({exc}); continuing without it.")
                self._tb = None

        if getattr(settings, "DQN_WANDB_ENABLE", False):
            try:
                import wandb

                cfg = {
                    "gamma": self.gamma,
                    "lr": self.lr,
                    "batch": self.batch,
                    "buffer": self.buffer_cap,
                    "epsilon_start": self.eps_start,
                    "epsilon_end": self.eps_end,
                    "epsilon_decay": self.eps_decay,
                    "period_steps": self.period_steps,
                    "move_step": self.move_step,
                    "sb3_target_update": self.sb3_target_update,
                }
                proj = getattr(settings, "DQN_WANDB_PROJECT", "ai-ran-dqn")
                name = getattr(settings, "DQN_WANDB_RUNNAME", "") or None
                self._wandb = wandb.init(project=proj, name=name, config=cfg)
                print(f"{self.xapp_id}: W&B logging enabled (project={proj})")
            except Exception as exc:
                print(f"{self.xapp_id}: W&B unavailable ({exc}); continuing without it.")
                self._wandb = None

        # Build a trivial Gym env so SB3 can instantiate its policy/Q-networks
        def _make_env():
            return _StaticPRBEnv(self._state_dim, self._n_actions)

        self._sb3_env = DummyVecEnv([_make_env])

        if self.model_arch != "mlp":
            print(
                f"{self.xapp_id}: Sequential arch '{self.model_arch}' requested but SB3 policy"
                " uses MlpPolicy; falling back to MLP."
            )
            self.model_arch = "mlp"

        policy_kwargs = {"net_arch": [256]}  # 18×256×7 (single hidden layer) per Tractor paper baseline
        self._model = SB3DQN(
            policy="MlpPolicy",
            env=self._sb3_env,
            learning_rate=self.lr,
            buffer_size=self.buffer_cap,
            learning_starts=0,
            batch_size=self.batch,
            gamma=self.gamma,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=max(1, self.sb3_target_update),
            exploration_fraction=1.0,  # we control epsilon manually
            exploration_initial_eps=self.eps_start,
            exploration_final_eps=self.eps_end,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=None,
        )

        try:
            from stable_baselines3.common.logger import configure

            if not hasattr(self._model, "_logger") or self._model._logger is None:  # type: ignore[attr-defined]
                self._model.set_logger(configure(folder=None, format_strings=[]))
        except Exception:
            pass

        # Try loading existing parameters
        try:
            if os.path.exists(self.model_path):
                self._model = SB3DQN.load(self.model_path, env=self._sb3_env)
                print(f"{self.xapp_id}: loaded SB3 model from {self.model_path}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to load SB3 model ({exc}); starting fresh.")
            self._model = SB3DQN(
                    policy="MlpPolicy",
                    env=self._sb3_env,
                    learning_rate=self.lr,
                    buffer_size=self.buffer_cap,
                    learning_starts=0,
                    batch_size=self.batch,
                    gamma=self.gamma,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=max(1, self.sb3_target_update),
                    exploration_fraction=1.0,
                    exploration_initial_eps=self.eps_start,
                    exploration_final_eps=self.eps_end,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    tensorboard_log=None,
                )
        try:
            from stable_baselines3.common.logger import configure

            if not hasattr(self._model, "_logger") or self._model._logger is None:  # type: ignore[attr-defined]
                self._model.set_logger(configure(folder=None, format_strings=[]))
        except Exception:
            pass

    # ---------------- Lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        mode = "train" if self.train_mode else "eval"
        print(f"{self.xapp_id}: enabled (mode={mode}, period={self.period_steps} steps)")

    # ---------------- Helper utilities (shared with custom DQN xApp) ----------------
    def _get_slice_counts(self, cell):
        cnt = {SL_E: 0, SL_U: 0, SL_M: 0}
        for ue in cell.connected_ue_list.values():
            s = getattr(ue, "slice_type", None)
            if s in cnt:
                cnt[s] += 1
        return cnt

    def _get_state(self, cell):
        cnt = self._get_slice_counts(cell)
        agg = self._aggregate_slice_metrics(cell)
        quota_raw = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        max_ue = MAX_UE_NORM
        max_prb = max(1.0, float(getattr(cell, "max_dl_prb", 1)))

        def clamp01(val: float) -> float:
            return max(0.0, min(1.0, float(val)))

        slice_order = (SL_M, SL_U, SL_E)
        state = []

        per_slice_debug = {}
        for sl in slice_order:
            raw = {
                "ue_count": float(cnt.get(sl, 0.0)),
                "prb_quota": float(agg.get(sl, {}).get("slice_prb", 0.0)),
                "tx_mbps": float(agg.get(sl, {}).get("tx_mbps", 0.0)),
                "buf_bytes": float(agg.get(sl, {}).get("buf_bytes", 0.0)),
                "prb_req": float(agg.get(sl, {}).get("prb_req", 0.0)),
                "prb_granted": float(agg.get(sl, {}).get("prb_granted", 0.0)),
            }
            per_slice_debug[sl] = {"raw": raw, "norm": {}}

        # UE load per slice (fraction of configured max UEs)
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["ue_count"] / max_ue)
            state.append(val)
            per_slice_debug[sl]["norm"]["ue_frac"] = val

        # Allocated PRB share per slice
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["prb_quota"] / max_prb)
            state.append(val)
            per_slice_debug[sl]["norm"]["prb_share"] = val

        # Served throughput (Mbps) normalised by configured cap
        dl_norm = max(1e-9, self.norm_dl_mbps)
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["tx_mbps"] / dl_norm)
            state.append(val)
            per_slice_debug[sl]["norm"]["throughput_norm"] = val

        # Buffer/backlog normalised by configured byte cap
        buf_norm = max(1e-9, self.norm_buf_bytes)
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["buf_bytes"] / buf_norm)
            state.append(val)
            per_slice_debug[sl]["norm"]["buffer_norm"] = val

        # PRB demand intensity as fraction of cell PRB budget
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["prb_req"] / max_prb)
            state.append(val)
            per_slice_debug[sl]["norm"]["prb_demand_norm"] = val

        # Grant satisfaction ratio per slice
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            req = raw["prb_req"]
            granted = raw["prb_granted"]
            if req <= 0.0:
                ratio = 1.0 if granted <= 0.0 else granted / max_prb
            else:
                ratio = granted / max(1.0, req)
            val = clamp01(ratio)
            state.append(val)
            per_slice_debug[sl]["norm"]["grant_ratio"] = val

        if getattr(settings, "DQN_PRB_DEBUG_STATE", False):
            interval = max(1, int(getattr(settings, "DQN_PRB_DEBUG_INTERVAL", 1)))
            decision_idx = getattr(self, "_t", 0)
            if decision_idx % interval == 0:
                debug_payload = {
                    "cell": getattr(cell, "cell_id", "unknown"),
                    "step": int(decision_idx),
                    "max_prb": float(max_prb),
                    "max_ue": float(max_ue),
                    "quota_raw": {k: float(v) for k, v in quota_raw.items()},
                    "per_slice": per_slice_debug,
                    "state": state,
                }
                try:
                    import json

                    print(f"{self.xapp_id}: STATE_DEBUG {json.dumps(debug_payload, default=float)}")
                except Exception:
                    print(f"{self.xapp_id}: STATE_DEBUG {debug_payload}")

        return state

    def _aggregate_slice_metrics(self, cell):
        quota_map = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        max_prb = float(getattr(cell, "max_dl_prb", 0) or 0.0)
        if SL_E not in quota_map:
            other = float(quota_map.get(SL_M, 0.0)) + float(quota_map.get(SL_U, 0.0))
            quota_map[SL_E] = max(0.0, max_prb - other)

        agg = {
            sl: {
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "prb_req": 0.0,
                "prb_granted": 0.0,
                "slice_prb": float(quota_map.get(sl, 0.0)),
            }
            for sl in (SL_E, SL_U, SL_M)
        }

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
            agg[sl]["prb_granted"] += float((alloc.get("downlink", 0.0) or 0.0))

        return agg

    def _slice_scores(self, cell, T_s):
        """Mirror the custom DQN reward shaping using available slice KPIs."""
        kappa = 8.0
        agg = self._aggregate_slice_metrics(cell)

        def clamp01(val: float) -> float:
            return max(0.0, min(1.0, float(val)))
        per_slice = {}
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
            per_slice[sl] = {
                "throughput": throughput_norm,
                "backlog": backlog_norm,
                "satisfaction": satisfaction,
                "need": need,
                "oversupply": oversupply,
                "idle": idle,
                "tx_mbps": tx_mbps,
                "buf_bytes": buf_bytes,
                "prb_req": demand,
                "prb_granted": granted,
            }
        embb = per_slice[SL_E]
        backlog_relief_e = 1.0 - embb["backlog"]
        embb_need_boost = 0.5 + 0.5 * embb["need"]
        embb_score = embb_need_boost * (
            0.55 * embb["satisfaction"]
            + 0.25 * embb["throughput"]
            + 0.20 * backlog_relief_e
        )
        embb_score -= 0.10 * embb["oversupply"]
        embb_score = clamp01(embb_score)
        
        urllc = per_slice[SL_U]
        buf_bits_u = urllc["buf_bytes"] * kappa
        tx_bps_u = urllc["tx_mbps"] * 1e6
        if tx_bps_u <= 0.0:
            delay_s = float("inf") if buf_bits_u > 0.0 else 0.0
        else:
            delay_s = buf_bits_u / max(1e-9, tx_bps_u)
        gamma = max(1e-12, float(self.urlc_gamma_s))
        if math.isfinite(delay_s):
            delay_term = math.exp(-delay_s / gamma)
            
        else:
            delay_term = 0.0
        delay_term = clamp01(delay_term)
        urllc_need_boost = 0.4 + 0.6 * urllc["need"]
        backlog_relief_u = 1.0 - urllc["backlog"]
        urllc_score = urllc_need_boost * (
            0.60 * delay_term
            + 0.30 * urllc["satisfaction"]
            + 0.10 * backlog_relief_u
        )
        urllc_score -= 0.05 * urllc["oversupply"]
        urllc_score = clamp01(urllc_score)
        mmtc = per_slice[SL_M]
        utilisation_m = 1.0 - mmtc["idle"]
        backlog_relief_m = 1.0 - mmtc["backlog"]
        mmtc_need_boost = 0.3 + 0.7 * mmtc["need"]
        mmtc_score = mmtc_need_boost * (
            0.50 * mmtc["satisfaction"]
            + 0.30 * utilisation_m
            + 0.20 * backlog_relief_m
        )
        mmtc_score -= 0.10 * mmtc["oversupply"]
        mmtc_score = clamp01(mmtc_score)
        return embb_score, urllc_score, mmtc_score

    def _reward(self, cell, T_s):
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _select_action(self, obs_vec: np.ndarray):
        if not self._model:
            return 0
        if self.train_mode:
            eps = _linear_eps(self.eps_start, self.eps_end, self.eps_decay, self._t)
            self._last_eps = eps
            if random.random() < eps:
                return int(self._sb3_env.action_space.sample())  # type: ignore[union-attr]
        action, _ = self._model.predict(obs_vec, deterministic=True)
        if hasattr(self._model, "exploration_rate"):
            try:
                self._last_eps = float(self._model.exploration_rate)  # type: ignore[attr-defined]
            except Exception:
                pass
        return int(action)

    def _apply_action(self, cell, action: int):
        combo = self._action_combos[int(action)]
        deltas = {
            SL_E: combo[0],
            SL_U: combo[1],
            SL_M: combo[2],
        }
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        for sl in (SL_E, SL_U, SL_M):
            quotas.setdefault(sl, 0)
        max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
        if max_prb <= 0:
            return False
        delta_prb = max(1, int(self.move_step))
        total = sum(quotas.values())
        changed = False
        for sl, sign in deltas.items():
            if sign < 0:
                reduce = min(delta_prb, quotas[sl])
                if reduce > 0:
                    quotas[sl] -= reduce
                    total -= reduce
                    changed = True
        for sl, sign in deltas.items():
            if sign > 0:
                available = max(0, max_prb - total)
                if available <= 0:
                    continue
                inc = min(delta_prb, available)
                if inc > 0:
                    quotas[sl] += inc
                    total += inc
                    changed = True
        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", None)
        moved_total = delta_prb * sum(abs(v) for v in deltas.values())
        if not changed:
            try:
                setattr(cell, "rl_last_action", {
                    "actor": "DQN",
                    "source": "SB3",
                    "code": int(action),
                    "label": self._action_labels[int(action)],
                    "changed": False,
                    "deltas": deltas,
                    "moved": 0,
                    "step": int(sim_step) if sim_step is not None else None,
                    "quota": dict(getattr(cell, "slice_dl_prb_quota", {}) or {}),
                })
            except Exception:
                pass
            return False
        cell.slice_dl_prb_quota = {s: int(max(0, quotas.get(s, 0))) for s in quotas}
        self._action_counts[action] += 1
        try:
            setattr(cell, "rl_last_action", {
                "actor": "DQN",
                "source": "SB3",
                "code": int(action),
                "label": self._action_labels[int(action)],
                "changed": True,
                "deltas": deltas,
                "moved": moved_total,
                "step": int(sim_step) if sim_step is not None else None,
                "quota": dict(cell.slice_dl_prb_quota),
            })
        except Exception:
            pass
        return True

    def _log(self, step_idx: int, cell_id: str, metrics: dict):
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(f"cell/{cell_id}/{k}", float(v), step_idx)
                except Exception:
                    pass
        if self._wandb is not None:
            try:
                import wandb

                self._wandb.log({f"cell/{cell_id}/{k}": v for k, v in metrics.items()}, step=step_idx)
            except Exception:
                pass

    # ---------------- Main control loop ----------------
    def step(self):
        if not self.enabled or not self._model:
            return

        sim_engine = getattr(self.ric, "simulation_engine", None)
        sim_step = getattr(sim_engine, "sim_step", 0)
        if sim_step % self.period_steps != 0:
            return

        dt = getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0)

        for cell in self.cell_list.values():
            obs = np.array(self._get_state(cell), dtype=np.float32)
            prev = self._per_cell_prev.get(cell.cell_id)

            if prev is not None and self.train_mode:
                reward = self._reward(cell, dt * self.period_steps)
                obs_prev = np.array(prev["obs"], dtype=np.float32)
                action_prev = int(prev["action"])

                obs_batch = obs_prev.reshape(1, -1)
                next_obs_batch = obs.reshape(1, -1)
                action_batch = np.array([[action_prev]], dtype=np.int64)
                reward_batch = np.array([reward], dtype=np.float32)
                done_batch = np.array([0.0], dtype=np.float32)

                try:
                    self._model.replay_buffer.add(
                        obs_batch,
                        next_obs_batch,
                        action_batch,
                        reward_batch,
                        done_batch,
                        infos=[{}],
                    )
                    if self._model.replay_buffer.size() >= max(32, self.batch):
                        self._model.train(gradient_steps=1, batch_size=self.batch)
                        self._last_loss = None
                    progress = max(0.0, 1.0 - self._t / max(1, self.sb3_total_steps))
                    self._model._current_progress_remaining = progress
                    self._model._on_step()
                except Exception as exc:
                    print(f"{self.xapp_id}: SB3 training error: {exc}")

            action = self._select_action(obs)
            self._apply_action(cell, action)
            self._per_cell_prev[cell.cell_id] = {"obs": obs, "action": action}

            # Per-step logging to TB/W&B
            try:
                T_s = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0)) * float(self.period_steps)
                reward_val = self._reward(cell, T_s)
                embb_score, urllc_score, mmtc_score = self._slice_scores(cell, T_s)
                prb_map = getattr(cell, "slice_dl_prb_quota", {}) or {}
                agg = self._aggregate_slice_metrics(cell)
                metrics = {
                    "reward": reward_val,
                    "embb_score": embb_score,
                    "urllc_score": urllc_score,
                    "mmtc_score": mmtc_score,
                    "epsilon": self._last_eps if self.train_mode else 0.0,
                    "loss": self._last_loss if self._last_loss is not None else 0.0,
                    "prb_eMBB": float(prb_map.get(SL_E, 0) or 0.0),
                    "prb_URLLC": float(prb_map.get(SL_U, 0) or 0.0),
                    "prb_mMTC": float(prb_map.get(SL_M, 0) or 0.0),
                    "embb_buf_bytes": float(agg[SL_E]["buf_bytes"]),
                    "urllc_buf_bytes": float(agg[SL_U]["buf_bytes"]),
                    "mmtc_buf_bytes": float(agg[SL_M]["buf_bytes"]),
                    "embb_tx_mbps": float(agg[SL_E]["tx_mbps"]),
                    "urllc_tx_mbps": float(agg[SL_U]["tx_mbps"]),
                    "mmtc_tx_mbps": float(agg[SL_M]["tx_mbps"]),
                }
                self._log(self._t, cell.cell_id, metrics)
            except Exception:
                pass

        self._t += 1

        if (
            self.train_mode
            and self.sb3_log_interval > 0
            and self._t % self.sb3_log_interval == 0
            and self._model is not None
        ):
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self._model.save(self.model_path)
                print(f"{self.xapp_id}: saved SB3 model to {self.model_path}")
            except Exception as exc:
                print(f"{self.xapp_id}: failed to save SB3 model ({exc})")

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "sb3_enabled": SB3_AVAILABLE,
                "train_mode": self.train_mode,
                "buffer_size": self.buffer_cap,
                "period_steps": self.period_steps,
                "move_step": self.move_step,
            }
        )
        return data
