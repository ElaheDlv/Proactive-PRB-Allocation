"""Gym-style PRB allocator xApp with episodic control over eMBB/URLLC slices.

This module implements a lightweight RL environment that mirrors the simulator
state through a Gym-like API (reset/step), plus a custom DQN agent that learns
to shift PRBs between two slices (eMBB, URLLC). Episodes are defined in a JSON
catalog that specifies UE counts, trace files, and initial PRB quotas.
"""

import json
import math
import os
import random
from collections import deque, defaultdict
from itertools import product
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

import settings
from utils import load_raw_packet_csv

from .xapp_base import xAppBase

SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    F = None


class _ReplayBuffer:
    """Simple FIFO replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        """Append a new transition to the buffer."""
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size):
        """Randomly sample a mini-batch (without replacement)."""
        batch_size = min(batch_size, len(self.buf))
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buf[i] for i in idx])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class _DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Two-hidden-layer MLP used when model_arch=mlp
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _LSTMDQN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # LSTM encodes the last seq_len states before feeding a small head
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        # x: [batch, seq_len, in_dim]
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.head(feat)


class PRBGymEnv:
    """Minimal Gym-like environment that exposes reset/step for PRB allocation."""

    def __init__(self, ric):
        """Initialise environment and pull tunables from global settings."""
        self.ric = ric
        self.sim_engine = getattr(ric, "simulation_engine", None)
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))
        self.norm_dl_mbps = float(getattr(settings, "DQN_NORM_MAX_DL_MBPS", 100.0))
        self.norm_buf_bytes = float(getattr(settings, "DQN_NORM_MAX_BUF_BYTES", 1e6))
        self.need_saturation = max(1e-6, float(getattr(settings, "DQN_NEED_SATURATION", 1.5)))
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.5))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.5))
        total_w = self.w_e + self.w_u
        if total_w <= 0:
            self.w_e = self.w_u = 0.5
            print(f"{self.xapp_id}: invalid slice weights; falling back to 0.5/0.5.")
        else:
            self.w_e /= total_w
            self.w_u /= total_w
        self._latency_targets = self._init_latency_targets()
        self._latency_spans = self._init_latency_spans()
        self._latency_bonus = self._init_latency_bonus()
        self._prb_penalty = self._init_prb_penalty()
        self._latency_sigmoid = self._init_latency_sigmoid()
        self._latency_tail = self._init_latency_tail()

        self._episodes = self._load_episode_specs()
        self._episode_loop = getattr(settings, "PRB_GYM_LOOP", False)
        self._episode_idx = -1
        self._current_episode = None
        self._steps_in_episode = 0
        self._episode_step_limit = 0
        self._managed_ues: set[str] = set()
        self._trace_cache: Dict[Tuple[str, str, float], List[Tuple[float, float, float]]] = {}
        self._action_combos = list(product([-1, 0, 1], repeat=2))
        self._state_dim = 8
        self._catalog_done = False
        self._episode_progress_bar = None
        self._progress_interval = max(1, int(getattr(settings, "PRB_GYM_PROGRESS_INTERVAL", 1000)))

    # ------------------------------------------------------------------ Episode control
    def _load_episode_specs(self) -> List[Dict]:
        """Load/normalise the episode catalog described in PRB_GYM_CONFIG_PATH."""
        path = getattr(settings, "PRB_GYM_CONFIG_PATH", "").strip()
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except Exception as exc:
            print(f"PRBGymEnv: failed to load episode config {path}: {exc}")
            return []
        if isinstance(raw, dict):
            entries = raw.get("episodes") or []
        elif isinstance(raw, list):
            entries = raw
        else:
            return []
        specs = []
        for idx, entry in enumerate(entries):
            spec = self._normalize_episode(entry, idx)
            if spec:
                specs.append(spec)
        return specs

    def _normalize_episode(self, entry, idx: int) -> Optional[Dict]:
        """Validate a raw episode dict; return None if it is incomplete."""
        if not isinstance(entry, dict):
            return None
        duration = int(entry.get("duration_steps", 0))
        if duration <= 0:
            print(f"PRBGymEnv: episode {idx} missing positive duration_steps; skipping.")
            return None
        slices = entry.get("slices") or {}
        e_cfg = self._normalize_slice_cfg(slices.get(SL_E) or slices.get("eMBB"))
        u_cfg = self._normalize_slice_cfg(slices.get(SL_U) or slices.get("URLLC"))
        if e_cfg is None or u_cfg is None:
            print(f"PRBGymEnv: episode {idx} missing slice configs; skipping.")
            return None
        initial_prb = entry.get("initial_prb", {})
        return {
            "id": entry.get("id") or f"episode_{idx}",
            "duration_steps": duration,
            "freeze_mobility": bool(entry.get("freeze_mobility", True)),
            "initial_prb": {
                SL_E: int(initial_prb.get(SL_E, initial_prb.get("eMBB", 0))),
                SL_U: int(initial_prb.get(SL_U, initial_prb.get("URLLC", 0))),
            },
            "slices": {
                SL_E: e_cfg,
                SL_U: u_cfg,
            },
        }

    def _normalize_slice_cfg(self, cfg) -> Optional[Dict]:
        """Ensure per-slice config contains UE count and optional trace metadata."""
        if not isinstance(cfg, dict):
            return None
        count = int(cfg.get("ue_count", 0))
        if count <= 0:
            return None
        trace_speed = float(cfg.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        trace_bin = float(cfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)))
        if getattr(settings, "TRACE_BIN_OVERRIDE", False):
            trace_bin = float(getattr(settings, "TRACE_BIN", trace_bin))
        return {
            "ue_count": count,
            "trace": cfg.get("trace"),
            "ue_ip": cfg.get("ue_ip"),
            "trace_speedup": trace_speed,
            "trace_bin": trace_bin,
        }

    def reset(self):
        """Reset the simulator to the next episode and return the initial observation."""
        if not self._episodes or self.sim_engine is None:
            return None
        if self._catalog_done:
            return None
        next_idx = self._episode_idx + 1
        if next_idx >= len(self._episodes):
            if not self._episode_loop:
                print("PRBGymEnv: episode catalog exhausted; stopping further resets.")
                self._catalog_done = True
                return None
            next_idx = 0
        self._episode_idx = next_idx
        self._current_episode = self._episodes[next_idx]
        self._episode_step_limit = self._current_episode["duration_steps"]
        self._steps_in_episode = 0
        self._deploy_episode(self._current_episode)
        total = len(self._episodes)
        print(f"PRBGymEnv: episode {self._episode_idx + 1}/{total} -> '{self._current_episode['id']}' ({self._episode_step_limit} decisions)")
        self._reset_progress_bar()
        return self._get_state()

    def _deploy_episode(self, spec: Dict):
        """Apply PRB quotas and spawn the scripted UE population for an episode."""
        self._clear_managed_ues()
        self._apply_initial_prb(spec.get("initial_prb", {}))
        for slice_name in (SL_E, SL_U):
            cfg = spec["slices"][slice_name]
            for idx in range(cfg["ue_count"]):
                imsi = f"GIMSI_{spec['id']}_{slice_name}_{idx}"
                self._register_episode_ue(imsi, slice_name, cfg)

    def _clear_managed_ues(self):
        """Remove every UE from the simulator to ensure a deterministic start state."""
        if self.sim_engine is None:
            return
        for imsi in list(self.sim_engine.ue_list.keys()):
            self.sim_engine.deregister_ue(imsi)
        self._managed_ues.clear()

    def _apply_initial_prb(self, prb_map: Dict[str, int]):
        """Set initial slice quotas on every cell, normalising to the cell budget."""
        if self.sim_engine is None:
            return
        for cell in self.sim_engine.cell_list.values():
            quotas = dict(cell.slice_dl_prb_quota)
            for sl, val in prb_map.items():
                quotas[sl] = max(0, int(val))
            total = sum(quotas.values())
            max_prb = max(1, int(getattr(cell, "max_dl_prb", 1)))
            if total > max_prb:
                scale = max_prb / float(total)
                for sl in quotas:
                    quotas[sl] = int(quotas[sl] * scale)
            cell.slice_dl_prb_quota = quotas

    def _register_episode_ue(self, imsi: str, slice_name: str, slice_cfg: Dict):
        """Spawn a scripted UE, optionally pinning mobility and attaching traces."""
        if self.sim_engine is None:
            return
        max_attempts = max(1, int(getattr(settings, "PRB_GYM_REGISTER_RETRIES", 10)))
        ok = False
        attempt = 0
        while attempt < max_attempts:
            ok = self.sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
            if ok:
                break
            attempt += 1
            # Give the simulator a moment (cells may still be initialising) before retrying.
            try:
                import time

                time.sleep(float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 0.5) or 0.5))
            except Exception:
                pass
        if not ok:
            print(f"{self.xapp_id}: unable to register {imsi} after {max_attempts} attempts; skipping.")
            return
        ue = self.sim_engine.ue_list.get(imsi)
        if ue is None:
            return
        if self._current_episode.get("freeze_mobility", True):
            ue.speed_mps = 0
            try:
                ue.set_target(ue.position_x, ue.position_y)
            except Exception:
                pass
        trace_path = slice_cfg.get("trace")
        if trace_path:
            self._attach_trace(ue, trace_path, slice_cfg)
        self._managed_ues.add(imsi)

    def _attach_trace(self, ue, path: str, slice_cfg: Dict):
        """Attach (and cache) a trace for a UE belonging to a slice."""
        cache_key = (path, slice_cfg.get("ue_ip") or "", float(slice_cfg.get("trace_bin", 1.0)))
        if cache_key not in self._trace_cache:
            try:
                samples = load_raw_packet_csv(
                    path,
                    ue_ip=slice_cfg.get("ue_ip"),
                    bin_s=slice_cfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)),
                    overhead_sub_bytes=int(getattr(settings, "TRACE_OVERHEAD_BYTES", 0)),
                )
            except Exception as exc:
                print(f"PRBGymEnv: failed to load trace {path}: {exc}")
                samples = []
            # Cache the parsed samples so we don't re-read/parse the same
            # trace file repeatedly (many UEs or repeated episodes). The
            # cache key includes path, UE IP and bin size so different
            # variants are stored separately.
            # Storing an empty list on failure prevents repeated load attempts
            # and noisy logging for the same missing/invalid trace. The
            # caller will check for an empty samples list and skip attaching
            # the trace when appropriate.
            self._trace_cache[cache_key] = samples
        # Retrieve cached samples (either freshly loaded above or present
        # from a previous call).
        samples = self._trace_cache[cache_key]
        if not samples:
            return
        speed = float(slice_cfg.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        bs = ue.current_bs
        try:
            if bs and hasattr(bs, "attach_dl_trace"):
                bs.attach_dl_trace(ue.ue_imsi, samples, speed)
            else:
                ue.attach_trace(samples, speed)
        except Exception as exc:
            print(f"PRBGymEnv: failed to attach trace {path} to {ue.ue_imsi}: {exc}")

    # ------------------------------------------------------------------ Observation / reward
    def _get_state(self):
        """Build the observation vector (8 elements) from the single training cell."""
        cell = self._target_cell()
        if cell is None:
            return np.zeros(self._state_dim, dtype=np.float32)
        agg = self._aggregate_slice_metrics(cell)
        state = []
        max_ue = float(getattr(settings, "UE_DEFAULT_MAX_COUNT", 50))
        max_prb = max(1.0, float(getattr(cell, "max_dl_prb", 1)))
        dl_norm = max(1e-9, self.norm_dl_mbps)
        buf_norm = max(1e-9, self.norm_buf_bytes)
        for sl in (SL_E, SL_U):
            data = agg[sl]
            state.extend([
                self._clamp01(data["ue_count"] / max_ue),
                self._clamp01(data["slice_prb"] / max_prb),
                self._clamp01(data["tx_mbps"] / dl_norm),
                self._clamp01(data["buf_bytes"] / buf_norm),
                ])
        return np.asarray(state, dtype=np.float32)

    def _aggregate_slice_metrics(self, cell):
        """Aggregate per-slice KPIs used for both the state vector and reward."""
        quota_map = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        agg = {
            SL_E: {
                "ue_count": 0,
                "slice_prb": float(quota_map.get(SL_E, 0.0)),
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "latency_sum": 0.0,
                "latency_count": 0,
            },
            SL_U: {
                "ue_count": 0,
                "slice_prb": float(quota_map.get(SL_U, 0.0)),
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "latency_sum": 0.0,
                "latency_count": 0,
            },
        }
        for ue in getattr(cell, "connected_ue_list", {}).values():
            sl = getattr(ue, "slice_type", None)
            if sl not in agg:
                continue
            agg[sl]["ue_count"] += 1
            agg[sl]["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
            agg[sl]["buf_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            agg[sl]["latency_sum"] += float(getattr(ue, "downlink_latency", 0.0) or 0.0)
            agg[sl]["latency_count"] += 1
        return agg

    def _reward(self, cell):
        """Latency-focused reward with URLLC emphasis and under-target bonuses."""
        agg = self._aggregate_slice_metrics(cell)
        max_prb = max(1.0, float(getattr(cell, "max_dl_prb", 1.0)))
        scores: Dict[str, Dict[str, float]] = {}
        weights = {SL_E: self.w_e, SL_U: self.w_u}
        weighted = 0.0
        bonus_total = 0.0
        for sl in (SL_E, SL_U):
            data = agg[sl]
            latency_avg = 0.0
            cnt = data.get("latency_count", 0)
            if cnt > 0:
                latency_avg = float(data.get("latency_sum", 0.0) or 0.0) / max(1, cnt)
            score = self._slice_score(latency_avg, sl)
            bonus = self._slice_bonus(latency_avg, sl)
            slice_prb = float(data.get("slice_prb", 0.0) or 0.0)
            prb_pen = self._slice_prb_penalty(
                slice_prb,
                sl,
                max_prb,
                latency_avg,
            )
            score = max(0.0, score - prb_pen)
            buf_bytes = float(data.get("buf_bytes", 0.0) or 0.0)
            tx_mbps = float(data.get("tx_mbps", 0.0) or 0.0)
            scores[sl] = {
                "score": score,
                "bonus": bonus,
                "prb_penalty": prb_pen,
                "prb_usage_norm": slice_prb / max_prb,
                "prb_usage_prb": slice_prb,
                "latency": latency_avg,
                "buf_bytes": buf_bytes,
                "tx_mbps": tx_mbps,
            }
            weighted += weights[sl] * score
            bonus_total += bonus
        total_reward = max(-1.0, min(1.0, weighted + bonus_total))
        return total_reward, scores

    def _init_latency_targets(self) -> Dict[str, float]:
        """Read per-slice latency budgets from settings (default to 10s / 1s)."""
        slices_cfg = getattr(settings, "NETWORK_SLICES", {})

        def _target(name: str, default: float) -> float:
            return float(slices_cfg.get(name, {}).get("latency_dl", default) or default)

        return {
            SL_E: _target(SL_E, 0.1),   # 100 ms default (seconds)
            SL_U: _target(SL_U, 0.001), # 1 ms default (seconds)
        }

    def _init_latency_spans(self) -> Dict[str, float]:
        """Define how far beyond the target latency each slice can go before score=0."""
        slices_cfg = getattr(settings, "NETWORK_SLICES", {})

        def _span(name: str, fallback_mult: float) -> float:
            cfg = slices_cfg.get(name, {})
            span_val = cfg.get("latency_span")
            if span_val is not None:
                return max(1e-6, float(span_val))
            target = self._latency_targets.get(name, 1.0)
            return max(1e-6, target * fallback_mult)

        return {
            SL_E: _span(SL_E, 1.0),   # eMBB tolerates up to ~2× target (200 ms)
            SL_U: _span(SL_U, 2.0),   # URLLC score hits zero quickly (~3 ms)
        }

    def _init_latency_bonus(self) -> Dict[str, float]:
        """Per-slice coefficients for rewarding sub-target latency."""
        return {
            SL_E: float(getattr(settings, "PRB_GYM_LAT_BONUS_EMBB", 0.02)),
            SL_U: float(getattr(settings, "PRB_GYM_LAT_BONUS_URLLC", 0.05)),
        }

    def _init_prb_penalty(self) -> Dict[str, float]:
        """Per-slice coefficients penalising large PRB quotas."""
        return {
            SL_E: float(getattr(settings, "PRB_GYM_PRB_PENALTY_EMBB", 0.01)),
            SL_U: float(getattr(settings, "PRB_GYM_PRB_PENALTY_URLLC", 0.02)),
        }

    def _init_latency_sigmoid(self) -> Dict[str, float]:
        """Slope controls for the logistic component (higher => sharper drop)."""
        return {
            SL_E: float(getattr(settings, "PRB_GYM_SIGMOID_ALPHA_EMBB", 4.0)),
            SL_U: float(getattr(settings, "PRB_GYM_SIGMOID_ALPHA_URLLC", 8.0)),
        }

    def _init_latency_tail(self) -> Dict[str, float]:
        """Tail parameters for heavy penalties once violations grow large."""
        return {
            SL_E: float(getattr(settings, "PRB_GYM_TAIL_C_EMBB", 0.5)),
            SL_U: float(getattr(settings, "PRB_GYM_TAIL_C_URLLC", 0.3)),
        }

    def _slice_score(self, latency_avg: float, slice_name: str) -> float:
        """Smooth logistic+tail score in [0,1] measuring latency adherence."""
        target = max(1e-9, self._latency_targets.get(slice_name, 1.0))
        span = max(1e-9, self._latency_spans.get(slice_name, target))
        x = (latency_avg - target) / span  # normalised violation around zero
        alpha = max(1e-6, self._latency_sigmoid.get(slice_name, 6.0))
        z = alpha * x
        z = max(-60.0, min(60.0, z))  # prevent overflow in exp()
        central = 1.0 / (1.0 + math.exp(z))  # smooth drop near target
        pos = max(0.0, x)
        tail_c = max(1e-6, self._latency_tail.get(slice_name, 0.5))
        tail = 1.0 / (1.0 + pos / tail_c)  # heavy penalties for large violations
        w = pos / (1.0 + pos)  # blend factor once we exceed the budget
        score = (1.0 - w) * central + w * tail
        eps = 1e-3
        return max(eps, min(1.0 - eps, score))

    def _slice_bonus(self, latency_avg: float, slice_name: str) -> float:
        """Additional reward for staying below the latency budget."""
        coeff = max(0.0, self._latency_bonus.get(slice_name, 0.0))
        if coeff <= 0.0:
            return 0.0
        target = max(1e-9, self._latency_targets.get(slice_name, 1.0))
        under = max(0.0, target - latency_avg)
        return coeff * (under / target)

    def _slice_prb_penalty(self, slice_prb: float, slice_name: str, max_prb: float, latency_avg: float) -> float:
        """Penalty applied only when latency already meets target, scaled quadratically by PRB usage."""
        coeff = max(0.0, self._prb_penalty.get(slice_name, 0.0))
        if coeff <= 0.0:
            return 0.0
        target = max(1e-9, self._latency_targets.get(slice_name, 1.0))
        if latency_avg > target:
            return 0.0
        usage = max(0.0, slice_prb) / max(1e-6, max_prb)
        return coeff * (usage ** 2)

    def _reset_progress_bar(self):
        self._episode_progress_bar = None
        self._log_progress(reset=True)

    def _log_progress(self, done=False, reset=False):
        if not self._episode_progress_bar:
            total = max(1, self._episode_step_limit)
            bar_length = 40
            self._episode_progress_bar = {
                "total": total,
                "bar_length": bar_length,
            }
        total = self._episode_progress_bar["total"]
        bar_length = self._episode_progress_bar["bar_length"]
        progress = min(total, self._steps_in_episode)
        filled = int(bar_length * progress / total)
        bar = "#" * filled + "-" * (bar_length - filled)
        percent = 100.0 * progress / total
        msg = f"Episode {self._episode_idx + 1}/{len(self._episodes)} [{bar}] {percent:5.1f}% ({progress}/{total})"
        if done:
            msg += " ✓"
        if reset:
            msg += " (reset)"
        print(msg)

    # ------------------------------------------------------------------ Step
    def step(self, action_idx: int):
        cell = self._target_cell()
        if cell is None:
            return self._get_state(), 0.0, True, {}
        self._apply_action(cell, action_idx)  # mutate slice quotas
        reward = self._reward(cell)           # compute shaped reward
        state = self._get_state()             # observe next state
        self._steps_in_episode += 1
        done = self._steps_in_episode >= self._episode_step_limit or self._traces_consumed()
        info = {"episode_id": self._current_episode["id"], "step": self._steps_in_episode}
        if self._steps_in_episode % self._progress_interval == 0 or done:
            self._log_progress(done)
        return state, reward, done, info

    def _apply_action(self, cell, action_idx: int):
        """Translate a discrete action into PRB adjustments for both slices."""
        combo = self._action_combos[int(action_idx)]
        delta_prb = self.move_step
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        for sl in (SL_E, SL_U):
            quotas.setdefault(sl, 0)
        max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
        if max_prb <= 0:
            return
        # Remove PRBs for negative moves first
        for sl, sign in zip((SL_E, SL_U), combo):
            if sign < 0:
                reduce = min(delta_prb * abs(sign), quotas[sl])
                quotas[sl] -= reduce
        # Apply positive moves
        for sl, sign in zip((SL_E, SL_U), combo):
            if sign > 0:
                quotas[sl] += delta_prb * sign
        total = sum(quotas.values())
        if total > max_prb:
            scale = max_prb / float(max(1, total))
            for sl in quotas:
                quotas[sl] = int(quotas[sl] * scale)
        cell.slice_dl_prb_quota = {k: int(max(0, v)) for k, v in quotas.items()}

    # ------------------------------------------------------------------ Helpers
    def _target_cell(self):
        """Return the single cell we train on (first entry in the simulator)."""
        if self.sim_engine is None or not self.sim_engine.cell_list:
            return None
        # Use the first cell for now (single-cell training scenario)
        return next(iter(self.sim_engine.cell_list.values()))

    def _traces_consumed(self) -> bool:
        """Return True if every managed UE has replayed its trace to completion."""
        cell = self._target_cell()
        if cell is None or not self._managed_ues:
            return False
        bs = getattr(cell, "base_station", None)
        replay = getattr(bs, "_dl_replay", None) if bs else None
        if not replay:
            return False
        for imsi in self._managed_ues:
            state = replay.get(imsi)
            if not state:
                continue
            samples = state.get("samples") or []
            idx = int(state.get("idx", 0))
            if samples and idx < len(samples):
                return False
        return True

    @staticmethod
    def _clamp01(val: float) -> float:
        return max(0.0, min(1.0, float(val)))

    @staticmethod
    def _grant_ratio(granted: float, requested: float, max_prb: float) -> float:
        if requested <= 0.0:
            return 1.0 if granted <= 0.0 else max(0.0, min(1.0, granted / max_prb))
        return max(0.0, min(1.0, granted / max(1.0, requested)))

class xAppGymPRBAllocator(xAppBase):
    """Standalone Gym-style DQN xApp (eMBB/URLLC, episodic).

    This xApp wraps the simulator inside a Gym-like environment (reset/step) and
    trains a DQN (MLP or LSTM) to move PRBs between two slices. It owns the
    entire training loop, so it does not rely on the legacy PRB allocator code.
    """

    def _make_config_signature(self) -> str:
        parts = [
            f"arch-{self.run_tag}",
            f"move-{self.env.move_step}",
            f"period-{self.env.period_steps}",
            f"batch-{self.batch}",
            f"lr-{self.lr:g}",
            f"gamma-{self.gamma:g}",
        ]
        parts.append("eps-reset")
        return "__".join(parts)

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = getattr(settings, "PRB_GYM_ENABLE", False)
        if not self.enabled:
            return
        if not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch is required; disabling.")
            self.enabled = False
            return

        self.env = PRBGymEnv(ric)
        if not self.env._episodes:
            print(f"{self.xapp_id}: no episodes configured; disable via --prb-gym-config.")
            self.enabled = False
            return
        self.state_dim = self.env._state_dim
        self.n_actions = len(self.env._action_combos)
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-3))
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 100000))
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.05))
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 20000))
        self.target_sync = max(1, int(getattr(settings, "DQN_PRB_TARGET_UPDATE", 200)))
        self.save_interval = int(getattr(settings, "DQN_PRB_SAVE_INTERVAL", 0))
        self.train_max_steps = max(0, int(getattr(settings, "DQN_PRB_MAX_TRAIN_STEPS", 0)))
        self.model_arch = getattr(settings, "DQN_PRB_MODEL_ARCH", "mlp").lower()
        self.seq_len = max(1, int(getattr(settings, "DQN_PRB_SEQ_LEN", 1)))
        self.seq_hidden = int(getattr(settings, "DQN_PRB_SEQ_HIDDEN", 128))
        self.is_lstm = self.model_arch == "lstm"
        arch_tag = "lstm" if self.is_lstm else "mlp"
        self.run_tag = f"{arch_tag}_seq{self.seq_len}" if self.is_lstm else "mlp"
        self._base_model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._config_signature = self._make_config_signature()
        base_root, base_ext = os.path.splitext(self._base_model_path)
        if not base_ext:
            base_ext = ".pt"
        base_dir = os.path.dirname(base_root) or "."
        base_name = os.path.basename(base_root) or "dqn_prb"
        model_dir = os.path.join(base_dir, self._config_signature)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, f"{base_name}_{self.run_tag}_{self._run_stamp}{base_ext}")

        self.device = self._select_device()
        self.q, self.q_target = self._build_models()
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.lr)
        self.replay = _ReplayBuffer(self.buffer_cap)
        self.timestep = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_done = False
        self._last_loss = None
        self._action_counts = defaultdict(int)
        self._tb = None
        self._tb_interval = max(1, int(getattr(settings, "DQN_PRB_LOG_INTERVAL", 1)))
        self._setup_tensorboard()
        self._state_window: deque = deque(maxlen=self.seq_len)
        self._episode_step_count = 0
        self._eps_decay_per_episode = max(1, int(getattr(settings, "PRB_GYM_EPS_DECAY_PER_EPISODE", 2000)))
        self._last_save_step = -1
        self._episode_return = 0.0
        self._episode_counter = 0
        self._reward_running_alpha = float(getattr(settings, "PRB_GYM_RUNNING_AVG_ALPHA", 0.01))
        self._reward_running_avg = 0.0
        self._reward_running_init = False
        self._cumulative_return = 0.0

    def start(self):
        """Reset the environment and print a short banner."""
        if not self.enabled:
            return
        raw_state = self.env.reset()
        if raw_state is None:
            print(f"{self.xapp_id}: no episodes available; disabling.")
            self.enabled = False
            return
        self.last_state = self._encode_state(raw_state, reset=True)
        self._episode_step_count = 0
        self._episode_return = 0.0
        self._cumulative_return = 0.0
        self._reward_running_avg = 0.0
        self._reward_running_init = False
        print(f"{self.xapp_id}: enabled (state_dim={self.state_dim}, actions={self.n_actions})")

    def step(self):
        """Main training loop hook invoked by the simulator every sim step."""
        if not self.enabled or self.last_state is None:
            return
        sim_engine = getattr(self.ric, "simulation_engine", None)
        if sim_engine is None:
            return
        sim_step = getattr(sim_engine, "sim_step", 0)
        # Only act every `period_steps` ticks to match radio scheduling cadence.
        if (self.env.period_steps > 1) and (sim_step % self.env.period_steps != 0):
            return
        if self.train_max_steps > 0 and self.timestep >= self.train_max_steps:
            self.enabled = False
            print(f"{self.xapp_id}: reached max train steps; shutting down.")
            self._maybe_save_checkpoint(force=True)
            return

        self.timestep += 1
        self._episode_step_count += 1
        action = self._select_action(self.last_state)
        self._action_counts[action] += 1
        next_state_raw, reward_info, done, _ = self.env.step(action)
        reward, slice_metrics = reward_info
        self._episode_return += reward
        self._cumulative_return += reward
        if not self._reward_running_init:
            self._reward_running_avg = reward
            self._reward_running_init = True
        else:
            alpha = self._reward_running_alpha
            self._reward_running_avg += alpha * (reward - self._reward_running_avg)
        next_state = self._encode_state(next_state_raw, reset=False)
        self.replay.push(self.last_state, action, reward, next_state, float(done))
        loss = self._optimize()
        if loss is not None:
            self._last_loss = loss
        if self._tb is not None and (self.timestep % self._tb_interval) == 0:
            self._log_tb(reward, self._epsilon(), self._last_loss, slice_metrics)
        if done:
            self._episode_counter += 1
            self._log_episode_return()
            self._episode_return = 0.0
            reset_obs = self.env.reset()
            if reset_obs is None:
                print(f"{self.xapp_id}: episode catalog finished; stopping training.")
                self._maybe_save_checkpoint(force=True)
                self.enabled = False
                return
            self.last_state = self._encode_state(reset_obs, reset=True)
            self._episode_step_count = 0
        else:
            self.last_state = next_state
        self.last_done = done

    # ------------------------------------------------------------------ DQN helpers
    def _select_action(self, state):
        """ε-greedy action selection."""
        eps = self._epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            if self.is_lstm:
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(s)
            return int(torch.argmax(q_vals, dim=1).item())

    def _epsilon(self):
        """Episode-aware epsilon decay schedule."""
        frac = min(1.0, self._episode_step_count / float(self._eps_decay_per_episode))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _optimize(self):
        """One SGD step on a sampled replay mini-batch."""
        if len(self.replay) < max(32, self.batch):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch)
        states = self._to_tensor(states)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = self._to_tensor(next_states)
        dones = dones.to(self.device)

        q_eval = self.q(states)
        q_pred = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_actions = self.q(next_states).max(1)[1]
            q_next = self.q_target(next_states).gather(1, online_actions.unsqueeze(1)).squeeze(1)
            q_target_val = rewards + (1.0 - dones) * self.gamma * q_next
        loss = F.smooth_l1_loss(q_pred, q_target_val)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.timestep % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self._maybe_save_checkpoint()
        return float(loss.item())

    def _select_device(self):
        """Choose runtime device based on DQN_PRB_DEVICE preference."""
        pref = (getattr(settings, "DQN_PRB_DEVICE", "auto") or "auto").lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(pref)
            print(f"{self.xapp_id}: requested {pref} but CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_models(self):
        """Instantiate policy and target networks (MLP or LSTM)."""
        if self.is_lstm:
            q = _LSTMDQN(self.state_dim, self.seq_hidden, self.n_actions).to(self.device)
            q_target = _LSTMDQN(self.state_dim, self.seq_hidden, self.n_actions).to(self.device)
        else:
            q = _DQN(self.state_dim, self.n_actions).to(self.device)
            q_target = _DQN(self.state_dim, self.n_actions).to(self.device)
        return q, q_target

    def _setup_tensorboard(self):
        """Initialise TensorBoard logging if enabled."""
        if not getattr(settings, "DQN_TB_ENABLE", False):
            return
        base = getattr(settings, "DQN_TB_DIR", "backend/tb_logs")
        logdir = os.path.join(base, f"prb_gym_{self.run_tag}_{self._run_stamp}")
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(logdir, exist_ok=True)
            self._tb = SummaryWriter(log_dir=logdir)
            print(f"{self.xapp_id}: TensorBoard logging to {logdir}")
        except Exception as exc:
            print(f"{self.xapp_id}: TensorBoard unavailable ({exc}); continuing without logging.")
            self._tb = None

    def _log_tb(self, reward: float, epsilon: float, loss: Optional[float], slice_metrics: Dict[str, dict]):
        """Write scalar metrics / histograms to TensorBoard."""
        if self._tb is None:
            return
        try:
            self._tb.add_scalar("train/reward", float(reward), self.timestep)
            self._tb.add_scalar("train/reward_running_avg", float(self._reward_running_avg), self.timestep)
            self._tb.add_scalar("train/return_cumulative", float(self._cumulative_return), self.timestep)
            self._tb.add_scalar("train/epsilon", float(epsilon), self.timestep)
            if loss is not None:
                self._tb.add_scalar("train/loss", float(loss), self.timestep)
            for sl, metrics in slice_metrics.items():
                label = "eMBB" if sl == SL_E else "URLLC"
                self._tb.add_scalar(f"train/{label}/score", float(metrics["score"]), self.timestep)
                if "bonus" in metrics:
                    self._tb.add_scalar(f"train/{label}/bonus", float(metrics["bonus"]), self.timestep)
                if "prb_penalty" in metrics:
                    self._tb.add_scalar(f"train/{label}/prb_penalty", float(metrics["prb_penalty"]), self.timestep)
                prb_norm = float(metrics.get("prb_usage_norm", 0.0))
                prb_abs = float(metrics.get("prb_usage_prb", 0.0))
                self._tb.add_scalar(f"train/{label}/prb_usage_norm", prb_norm, self.timestep)
                self._tb.add_scalar(f"train/{label}/prb_usage_prb", prb_abs, self.timestep)
                self._tb.add_scalar(f"train/{label}/latency", float(metrics["latency"]), self.timestep)
                self._tb.add_scalar(f"train/{label}/buffer_bytes", float(metrics["buf_bytes"]), self.timestep)
                self._tb.add_scalar(f"train/{label}/tx_mbps", float(metrics["tx_mbps"]), self.timestep)
            if self.timestep % (self._tb_interval * 10) == 0:
                counts = [self._action_counts.get(i, 0) for i in range(self.n_actions)]
                self._tb.add_histogram("actions", torch.tensor(counts, dtype=torch.float32), self.timestep)
        except Exception:
            pass

    def _log_episode_return(self):
        """Log cumulative reward per episode for learning diagnostics."""
        if self._tb is None:
            return
        try:
            self._tb.add_scalar("train/episode_return", float(self._episode_return), self._episode_counter)
        except Exception:
            pass

    def _checkpoint_path(self, step: Optional[int] = None) -> str:
        """Return checkpoint file path, optionally annotated with a step index."""
        base, ext = os.path.splitext(self.model_path)
        if not ext:
            base = self.model_path
        suffix = "_final" if step is None else f"_step{int(step)}"
        if ext:
            return f"{base}{suffix}{ext}"
        return f"{base}{suffix}"

    def _save_model(self, path: str):
        """Persist the online Q-network weights to disk."""
        if not TORCH_AVAILABLE or not path or getattr(self, "q", None) is None:
            return
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass
        try:
            torch.save(self.q.state_dict(), path)
        except Exception:
            pass

    def _maybe_save_checkpoint(self, force: bool = False):
        """Save checkpoints periodically or when forced (e.g. shutdown)."""
        if not TORCH_AVAILABLE or getattr(self, "q", None) is None:
            return
        if force:
            final_path = self._checkpoint_path(None)
            self._save_model(final_path)
            if self.timestep > 0:
                step_path = self._checkpoint_path(self.timestep)
                self._save_model(step_path)
            return
        if self.save_interval <= 0:
            return
        if self.timestep <= 0 or self.timestep == self._last_save_step:
            return
        if (self.timestep % self.save_interval) != 0:
            return
        final_path = self._checkpoint_path(None)
        self._save_model(final_path)
        step_path = self._checkpoint_path(self.timestep)
        self._save_model(step_path)
        self._last_save_step = self.timestep

    def __del__(self):
        """Best-effort cleanup for the TB writer."""
        try:
            self._maybe_save_checkpoint(force=True)
            if self._tb is not None:
                self._tb.flush()
                self._tb.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ State encoding helpers
    def _encode_state(self, raw_state, reset: bool = False):
        """Return either the raw vector (MLP) or a `[seq_len, dim]` LSTM window."""
        arr = np.asarray(raw_state, dtype=np.float32)
        if not self.is_lstm:
            return arr
        if reset or self.timestep == 0:
            self._state_window.clear()  # restart history at each episode boundary
        self._state_window.append(arr)  # push most recent observation
        seq = np.zeros((self.seq_len, self.state_dim), dtype=np.float32)
        window = list(self._state_window)
        seq[-len(window):] = window
        return seq

    def _to_tensor(self, arr):
        """Move state batches to the configured device."""
        tensor = torch.tensor(arr, dtype=torch.float32)
        if not self.is_lstm:
            return tensor.to(self.device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
