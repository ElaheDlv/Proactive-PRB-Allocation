"""DQN-based PRB allocator xApp (with extensive inline documentation).

This xApp implements a light Deep Q-Network policy that shifts downlink PRBs
between slices in each cell. It closely follows the Tractor paper’s MDP.

- State (per cell): UE mix, PRB share, throughput, buffer backlog, PRB demand,
  and grant satisfaction ratios for mMTC/URLLC/eMBB (all normalised to ~[0,1]).
- Actions: each slice (eMBB/URLLC/mMTC) receives a {-1, 0, +1} multiplier that
           indicates decrease, keep, or increase by a fixed step (``DQN_PRB_MOVE_STEP``),
           yielding 27 discrete actions in total.
- Reward: weighted sum of slice‑specific scores (eMBB queue drain proxy,
  URLLC delay proxy, mMTC utilization/idle penalty).

Training support:
- Online training with epsilon‑greedy exploration, replay buffer, target network.
- Optional telemetry: TensorBoard and Weights & Biases for metrics.

All key methods include detailed comments to make the logic easy to follow.
"""

from .xapp_base import xAppBase

import os
import math
import random
from collections import deque, defaultdict
from datetime import datetime
from typing import Optional
from itertools import product

import settings  # global configuration and constants

import numpy as np

from .seq_models import LSTMModel, TemporalConvNet, Seq2SeqAttentionModel

try:  # torch is optional; the xApp disables itself if not available
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    # Do not crash if torch is missing; xApp will disable itself.
    TORCH_AVAILABLE = False
    print("xapp_dqn_prb_allocator: torch not available; xApp will disable itself.")


# Short aliases for slice names (read from settings with fallbacks)
SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")   # Enhanced Mobile Broadband
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")  # Ultra Reliable Low Latency
SL_M = getattr(settings, "NETWORK_SLICE_MTC_NAME", "mMTC")     # Massive Machine Type


if TORCH_AVAILABLE:
    class _ReplayBuffer:
        """Minimal replay buffer for off‑policy training.

        Stores tuples (state, action, reward, next_state, done) in a ring buffer
        and supports random mini‑batch sampling.
        """
        def __init__(self, capacity: int = 50000):
            self.buf = deque(maxlen=int(capacity))  # fixed‑size circular buffer

        def push(self, s, a, r, ns, d):
            self.buf.append((s, a, r, ns, d))

        def sample(self, batch):
            import numpy as np

            batch = min(batch, len(self.buf))  # cap to current buffer size
            idx = np.random.choice(len(self.buf), batch, replace=False)  # unique indices
            s, a, r, ns, d = zip(*[self.buf[i] for i in idx])
            return (
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(ns, dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
            )

        def __len__(self):
            return len(self.buf)


    class _DQN(nn.Module):
        """Small fully‑connected Q‑network.

        Architecture: single hidden layer (256 units, ReLU) → Q‑values.
        """
        def __init__(self, in_dim: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, n_actions),
            )

        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]
            return self.net(x)


    class _SeqQNetwork(nn.Module):
        """Wrapper around sequence models to produce Q-values."""

        def __init__(self, in_dim: int, n_actions: int, arch: str, hidden_dim: int = 128):
            super().__init__()
            arch = arch.lower()
            if arch == "lstm":
                self.model = LSTMModel(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    output_dim=n_actions,
                )
            elif arch == "tcn":
                self.model = TemporalConvNet(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    output_dim=n_actions,
                )
            elif arch in ("seq2seq", "seq2seq-attn", "seq2seq_attention"):
                self.model = Seq2SeqAttentionModel(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    output_dim=n_actions,
                )
            else:
                raise ValueError(f"Unsupported sequential architecture: {arch}")

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.model(x)
else:
    # Placeholders to avoid NameError if referenced in disabled code paths
    _ReplayBuffer = None
    _DQN = None


class xAppDQNPRBAllocator_simple_past(xAppBase):
    """DQN-based PRB allocator (Table 3 actions).

    Public methods:
    - start(): lifecycle hook when xApp is loaded
    - step(): called every simulation step; performs observe→learn→act
    - to_json(): returns metadata for introspection

    Private helpers:
    - _epsilon(): exploration schedule
    - _get_state(): builds the per‑cell state vector
    - _aggregate_slice_metrics(): aggregates KPIs needed for rewards
    - _slice_scores(): computes eMBB/URLLC/mMTC slice scores
    - _reward(): combines slice scores with weights
    - _act(): epsilon‑greedy action selection
    - _opt_step(): one optimization step on a replay mini‑batch
    - _apply_action(): applies PRB move to cell quotas
    - _log(): emits metrics to TB/W&B
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = getattr(settings, "DQN_PRB_ENABLE", False)  # on/off toggle

        # Runtime / training knobs
        self.train_mode = getattr(settings, "DQN_PRB_TRAIN", True)  # train or inference only
        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))  # act every N steps
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))  # PRBs moved per action
        self.norm_dl_mbps = float(getattr(settings, "DQN_NORM_MAX_DL_MBPS", 100.0))  # throughput normaliser
        self.norm_buf_bytes = float(getattr(settings, "DQN_NORM_MAX_BUF_BYTES", 1e6))  # buffer normaliser
        self.need_saturation = max(1e-6, float(getattr(settings, "DQN_NEED_SATURATION", 1.5)))  # demand-to-quota cap

        # Reward shaping/weights
        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.33))   # weight for eMBB slice score
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.34))  # weight for URLLC slice score
        self.w_m = float(getattr(settings, "DQN_WEIGHT_MMTC", 0.33))   # weight for mMTC slice score
        self.urlc_gamma_s = float(getattr(settings, "DQN_URLLC_GAMMA_S", 0.01))  # max tolerable delay (s)

        # DQN parameters
        self.gamma = float(getattr(settings, "DQN_PRB_GAMMA", 0.99))        # discount factor
        self.lr = float(getattr(settings, "DQN_PRB_LR", 1e-2))              # learning rate
        self.batch = int(getattr(settings, "DQN_PRB_BATCH", 64))             # mini‑batch size
        self.buffer_cap = int(getattr(settings, "DQN_PRB_BUFFER", 50000))    # replay capacity
        self.eps_start = float(getattr(settings, "DQN_PRB_EPSILON_START", 1.0))  # ε start
        self.eps_end = float(getattr(settings, "DQN_PRB_EPSILON_END", 0.1))      # ε end
        self.eps_decay = int(getattr(settings, "DQN_PRB_EPSILON_DECAY", 10000))  # ε decay steps
        self._base_model_path = getattr(settings, "DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")  # checkpoint
        self.model_path = self._base_model_path
        self.save_interval = int(getattr(settings, "DQN_PRB_SAVE_INTERVAL", 0) or 0)
        if self.save_interval < 0:
            self.save_interval = 0

        # Internal state
        self._t = 0  # global decision counter (used for ε schedule and logging)
        self._per_cell_prev = {}  # cell_id -> {state, action} for previous decision
        self._last_loss = None    # last training loss (for TB/W&B)
        self._action_counts = defaultdict(int)  # histogram of actions taken
        self.seq_len = max(1, int(getattr(settings, "DQN_PRB_SEQ_LEN", 1)))
        self.seq_hidden_dim = int(getattr(settings, "DQN_PRB_SEQ_HIDDEN", 128))
        self._state_history = defaultdict(lambda: deque(maxlen=self.seq_len))
        self._reward_ema: Optional[float] = None
        self._reward_ema_alpha = float(getattr(settings, "DQN_PRB_REWARD_EMA_ALPHA", 0.01))
        if not (0.0 < self._reward_ema_alpha <= 1.0):
            self._reward_ema_alpha = 0.01
        self._slice_metrics_to_track = (
            "satisfaction",
            "throughput",
            "backlog",
            "oversupply",
            "idle",
            "need",
            "tx_mbps",
            "buf_bytes",
            "prb_req",
            "prb_granted",
        )
        self._slice_metrics_avg = {
            "satisfaction",
            "throughput",
            "backlog",
            "oversupply",
            "idle",
            "need",
        }
        self._slice_metrics_sum = {
            "tx_mbps",
            "buf_bytes",
            "prb_req",
            "prb_granted",
        }
        # Optional pseudo-episode windows for logging / bootstrapping control
        self.window_len = max(0, int(getattr(settings, "DQN_PRB_EPISODE_LEN", 0)))
        self.window_mark_done = bool(getattr(settings, "DQN_PRB_EPISODE_MARK_DONE", True))
        self._window_step = 0          # steps elapsed inside current window
        self._window_idx = 0           # completed window counter
        self._window_reward = 0.0      # cumulative reward within window
        self._window_loss_sum = 0.0    # aggregate loss for averaging
        self._window_loss_count = 0    # number of loss samples gathered
        self._window_cell_sums = {}    # cell_id -> slice -> metric -> sum
        self._window_cell_counts = defaultdict(int)
        self._window_cell_score_sums = {}  # cell_id -> slice -> score sum
        self._last_save_step = -1
        self._runtime_device = "cpu"
        self._scalar_log_interval = max(1, int(getattr(settings, "DQN_PRB_LOG_INTERVAL", 1)))

        # NN / action space
        self._action_combos = list(product([-1, 0, 1], repeat=3))
        self._action_labels = [
            f"Δe={combo[0]},Δu={combo[1]},Δm={combo[2]}"
            for combo in self._action_combos
        ]
        self._n_actions = len(self._action_combos)
        self._state_dim = 18  # length of the state vector
        self.model_arch = getattr(settings, "DQN_PRB_MODEL_ARCH", "mlp").lower()
        self._device = None
        if TORCH_AVAILABLE:
            self._device = self._select_device()
        if self.enabled and not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False

        # Telemetry: TensorBoard / W&B
        self._tb = None     # TensorBoard SummaryWriter (optional)
        self._wandb = None  # Weights & Biases run object (optional)
        if self.enabled:
            arch = self.model_arch
            if arch == "mlp":
                self._q = _DQN(self._state_dim, self._n_actions).to(self._device)
                self._q_target = _DQN(self._state_dim, self._n_actions).to(self._device)
                self.arch_tag = "mlp"
            else:
                if self.seq_len <= 1:
                    self.seq_len = max(2, self.seq_len)
                self._q = _SeqQNetwork(
                    self._state_dim,
                    self._n_actions,
                    arch,
                    hidden_dim=self.seq_hidden_dim,
                ).to(self._device)
                self._q_target = _SeqQNetwork(
                    self._state_dim,
                    self._n_actions,
                    arch,
                    hidden_dim=self.seq_hidden_dim,
                ).to(self._device)
                self.arch_tag = f"{arch}_seq{self.seq_len}"
                print(f"{self.xapp_id}: using {self.arch_tag.upper()} sequential extractor.")
            if arch == "mlp":
                self.arch_tag = "mlp"

            base, ext = os.path.splitext(self._base_model_path)
            if ext:
                self.model_path = f"{base}_{self.arch_tag}{ext}"
            else:
                self.model_path = f"{self._base_model_path}_{self.arch_tag}"

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self._q_target.load_state_dict(self._q.state_dict())
            self._opt = optim.Adam(self._q.parameters(), lr=self.lr)  # optimizer
            self._buf = _ReplayBuffer(self.buffer_cap)                 # replay buffer
            # Try to load existing model
            try:
                candidates = [self.model_path]
                if self.model_path != self._base_model_path:
                    candidates.append(self._base_model_path)
                for cand in candidates:
                    if not os.path.exists(cand):
                        continue
                    self._q.load_state_dict(torch.load(cand, map_location=self._device))
                    self._q_target.load_state_dict(self._q.state_dict())
                    self.model_path = cand
                    print(f"{self.xapp_id}: loaded model from {cand}")
                    break
            except Exception as e:
                print(f"{self.xapp_id}: failed loading model: {e}")

            # TensorBoard
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_stamp = run_stamp
            if getattr(settings, "DQN_TB_ENABLE", False):  # TensorBoard logging (optional)
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    base = getattr(settings, "DQN_TB_DIR", "backend/tb_logs")
                    logdir = os.path.join(base, f"dqn_prb_{self.arch_tag}_{run_stamp}")
                    os.makedirs(logdir, exist_ok=True)
                    # SummaryWriter expects 'log_dir', not 'logdir'
                    self._tb = SummaryWriter(log_dir=logdir)
                    print(f"{self.xapp_id}: TensorBoard logging to {logdir}")
                except Exception as e:
                    print(f"{self.xapp_id}: TensorBoard unavailable: {e}")
                    self._tb = None
            # W&B
            if getattr(settings, "DQN_WANDB_ENABLE", False):  # W&B logging (optional)
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
                        "model_arch": self.arch_tag,
                        "seq_len": self.seq_len,
                        "seq_hidden": self.seq_hidden_dim,
                    }
                    proj = getattr(settings, "DQN_WANDB_PROJECT", "ai-ran-dqn")
                    name = getattr(settings, "DQN_WANDB_RUNNAME", "") or None
                    if name is None:
                        name = f"{self.arch_tag}_{run_stamp}"
                    self._wandb = wandb.init(project=proj, name=name, config=cfg)
                    print(f"{self.xapp_id}: W&B logging enabled (project={proj})")
                except Exception as e:
                    print(f"{self.xapp_id}: W&B unavailable: {e}")
                    self._wandb = None

    # ---------------- xApp lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        print(f"{self.xapp_id}: enabled (train={self.train_mode}, period={self.period_steps} steps)")

    def _epsilon(self):
        """Return current exploration epsilon based on linear decay schedule."""
        if self.eps_decay <= 0:
            return self.eps_end
        frac = min(1.0, self._t / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _get_slice_counts(self, cell):
        """Count UEs per slice in a cell.

        Returns a dict {slice_name: count} for the three slices.
        """
        cnt = {SL_E: 0, SL_U: 0, SL_M: 0}
        for ue in cell.connected_ue_list.values():
            s = getattr(ue, "slice_type", None)
            if s in cnt:
                cnt[s] += 1
        return cnt

    def _get_state(self, cell):
        """Build normalized state vector for a cell.

        The vector includes per-slice load, resource allocation, traffic demand,
        and QoS proxies so both plain and LSTM agents observe identical KPIs.
        """
        cnt = self._get_slice_counts(cell)
        agg = self._aggregate_slice_metrics(cell)
        quota_raw = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        max_ue = max(1.0, float(getattr(settings, "UE_DEFAULT_MAX_COUNT", 50)))
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

        # UE mix per slice
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

        # Served throughput
        dl_norm = max(1e-9, self.norm_dl_mbps)
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["tx_mbps"] / dl_norm)
            state.append(val)
            per_slice_debug[sl]["norm"]["throughput_norm"] = val

        # Buffer backlog
        buf_norm = max(1e-9, self.norm_buf_bytes)
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["buf_bytes"] / buf_norm)
            state.append(val)
            per_slice_debug[sl]["norm"]["buffer_norm"] = val

        # PRB demand intensity
        for sl in slice_order:
            raw = per_slice_debug[sl]["raw"]
            val = clamp01(raw["prb_req"] / max_prb)
            state.append(val)
            per_slice_debug[sl]["norm"]["prb_demand_norm"] = val

        # Grant satisfaction ratio
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

    def _update_state_sequence(self, cell_id: str, state_vec):
        hist = self._state_history[cell_id]
        hist.append(np.array(state_vec, dtype=np.float32))
        seq = np.zeros((self.seq_len, self._state_dim), dtype=np.float32)
        h_list = list(hist)
        if h_list:
            seq[-len(h_list):] = h_list
        return seq

    def _aggregate_slice_metrics(self, cell):
        """Return per-slice aggregates used to compute rewards and state."""

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

    def _slice_scores(self, cell, T_s, return_components: bool = False):
        """Return per-slice scores that reward meeting demand with minimal waste.

        The new shaping aligns the reward with observable slice KPIs:
        - eMBB: favour high throughput and drained buffers when demand pressure is high.
        - URLLC: reward low queueing delay and reliability, scaled by latency demand.
        - mMTC: encourage efficient use of the reserved PRBs while avoiding over-allocation.
        """
        kappa = 8.0  # bits per byte
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

        scores = (embb_score, urllc_score, mmtc_score)
        if return_components:
            return scores, per_slice
        return scores
    
    def _reward(self, cell, T_s):
        """Combine slice scores using configured weights to a scalar reward."""
        e, u, m = self._slice_scores(cell, T_s)
        return float(self.w_e * e + self.w_u * u + self.w_m * m)

    def _act(self, state):
        """Epsilon‑greedy action selection (random with prob ε; otherwise argmax Q)."""
        eps = self._epsilon() if self.train_mode else 0.0
        if random.random() < eps or not TORCH_AVAILABLE:
            return random.randrange(self._n_actions)
        with torch.no_grad():
            device = self._device if self._device is not None else torch.device("cpu")
            x = torch.tensor([state], dtype=torch.float32, device=device)
            q = self._q(x)
            return int(torch.argmax(q, dim=1).item())

    def _select_device(self):
        """Select computation device based on availability and configuration."""
        pref = getattr(settings, "DQN_PRB_DEVICE", "auto") or "auto"
        pref_str = str(pref).strip()
        pref_lower = pref_str.lower()
        if pref_lower in ("auto", "default", ""):
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        try:
            device = torch.device(pref_str)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return device
        except Exception as exc:
            print(f"{self.xapp_id}: requested device '{pref_str}' unavailable ({exc}); falling back to CPU.")
            return torch.device("cpu")

    def _opt_step(self):  #### OPTIMIZATION STEP is so simple as of now we might need to add more features
        """One DQN optimization step over a replay mini‑batch.

        Returns the scalar loss value (float) when training occurs; otherwise None.
        """
        if not self.train_mode or not TORCH_AVAILABLE:
            return None
        if len(self._buf) < max(32, self.batch):
            return None
        device = self._device if self._device is not None else torch.device("cpu")
        s, a, r, ns, d = self._buf.sample(self.batch)
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        ns = ns.to(device)
        d = d.to(device)
        q = self._q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            qn = self._q_target(ns).max(1)[0]
            tgt = r + (1.0 - d) * self.gamma * qn
        loss = (q - tgt).pow(2).mean()
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        # Periodically refresh target network
        if self._t % 200 == 0:
            self._q_target.load_state_dict(self._q.state_dict())
        self._maybe_save_checkpoint()
        return float(loss.item())

    def _checkpoint_path(self, step: Optional[int] = None) -> str:
        """Return the checkpoint path, optionally suffixed with a step index."""
        if step is None:
            return self.model_path
        base, ext = os.path.splitext(self.model_path)
        suffix = f"_step{int(step)}"
        if ext:
            return f"{base}{suffix}{ext}"
        return f"{self.model_path}{suffix}"

    def _save_model(self, path: str):
        """Persist the current Q-network weights to disk."""
        if not TORCH_AVAILABLE or not path or not hasattr(self, "_q") or self._q is None:
            return
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass
        try:
            torch.save(self._q.state_dict(), path)
        except Exception:
            pass

    def _maybe_save_checkpoint(self):
        """Save model checkpoints on the configured interval."""
        if not TORCH_AVAILABLE or self.save_interval <= 0:
            return
        if self._t <= 0 or self._t == self._last_save_step:
            return
        if (self._t % self.save_interval) != 0:
            return
        self._save_model(self.model_path)
        step_path = self._checkpoint_path(self._t)
        if step_path != self.model_path:
            self._save_model(step_path)
        self._last_save_step = self._t

    def _apply_action(self, cell, action: int):
        """Apply a discrete action adjusting each slice by {-1,0,1}×move_step PRBs."""
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
        initial_quotas = {sl: int(quotas.get(sl, 0)) for sl in (SL_E, SL_U, SL_M)}
        total = sum(quotas.values())
        # Process decreases first to free up capacity
        for sl, sign in deltas.items():
            if sign < 0:
                reduce = min(delta_prb * abs(sign), quotas[sl])
                if reduce > 0:
                    quotas[sl] -= reduce
                    total -= reduce
        # Build desired totals after applying positive moves
        requested = {sl: delta_prb * sign for sl, sign in deltas.items() if sign > 0}
        desired_totals = {sl: float(quotas[sl]) for sl in (SL_E, SL_U, SL_M)}
        for sl, inc in requested.items():
            desired_totals[sl] = desired_totals.get(sl, 0.0) + float(inc)
        desired_sum = sum(desired_totals.values())
        final_quotas = {sl: int(max(0, quotas.get(sl, 0))) for sl in (SL_E, SL_U, SL_M)}
        if requested:
            if desired_sum <= max_prb:
                final_quotas = {sl: int(max(0, desired_totals.get(sl, 0.0))) for sl in (SL_E, SL_U, SL_M)}
            elif desired_sum > 0:
                scale = max_prb / float(desired_sum)
                fractional = []
                assigned = 0
                scaled = {}
                for sl in (SL_E, SL_U, SL_M):
                    raw = desired_totals.get(sl, 0.0) * scale
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
                final_quotas = {sl: int(max(0, scaled.get(sl, 0))) for sl in (SL_E, SL_U, SL_M)}
        quotas = final_quotas
        applied_deltas = {sl: quotas[sl] - initial_quotas.get(sl, 0) for sl in (SL_E, SL_U, SL_M)}
        changed = any(delta != 0 for delta in applied_deltas.values())

        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", None)
        moved_total = sum(abs(v) for v in applied_deltas.values())
        if not changed:
            # Log the attempted action even if it results in no change
            try:
                setattr(cell, "rl_last_action", {
                    "actor": "DQN",
                    "code": int(action),
                    "label": self._action_labels[int(action)],
                    "changed": False,
                    "deltas": deltas,
                    "applied": applied_deltas,
                    "moved": 0,
                    "step": int(sim_step) if sim_step is not None else None,
                    "quota": dict(getattr(cell, "slice_dl_prb_quota", {}) or {}),
                })
            except Exception:
                pass
            return False
        # Commit updated quotas
        cell.slice_dl_prb_quota = {s: int(max(0, quotas.get(s, 0))) for s in quotas}
        self._action_counts[action] += 1
        try:
            setattr(cell, "rl_last_action", {
                "actor": "DQN",
                "code": int(action),
                "label": self._action_labels[int(action)],
                "changed": True,
                "deltas": deltas,
                "applied": applied_deltas,
                "moved": moved_total,
                "step": int(sim_step) if sim_step is not None else None,
                "quota": dict(cell.slice_dl_prb_quota),
            })
        except Exception:
            pass
        return True
    
    def _log(self, step_idx: int, cell_id: str, metrics: dict):
        """Emit metrics to TensorBoard and/or Weights & Biases."""
        if self._tb is None and self._wandb is None:
            return
        if (step_idx % self._scalar_log_interval) != 0:
            return
        # TensorBoard scalars per cell
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(f"cell/{cell_id}/{k}", float(v), step_idx)
                except Exception:
                    pass
        # W&B
        if self._wandb is not None:
            try:
                import wandb
                self._wandb.log({f"cell/{cell_id}/{k}": v for k, v in metrics.items()}, step=step_idx)
            except Exception:
                pass

    def step(self):
        """Main loop: observe→(learn)→act at the configured decision period.

        - Early return on non‑decision steps to reduce overhead.
        - For each cell: compute new state s_t; if an (s_{t-1}, a_{t-1}) exists,
          compute reward r_t and push a transition (s_{t-1}, a_{t-1}, r_t, s_t).
        - Optionally optimize the network, then select and apply the next action.
        - Log per‑cell metrics and a periodic action histogram.
        """
        if not self.enabled:
            return
        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", 0)
        if sim_step % self.period_steps != 0:
            return
        self._t += 1
        T_s = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1)) * float(self.period_steps)
        window_terminal = self.window_len > 0 and (self._window_step + 1) >= self.window_len
        done_flag = 1.0 if (window_terminal and self.window_mark_done) else 0.0

        for cell_id, cell in self.cell_list.items():
            # Compute state now (after environment step allocated PRBs)
            s_now = self._get_state(cell)
            seq_now = self._update_state_sequence(cell_id, s_now)

            # If we have a pending (s,a) from previous decision, compute reward and push transition
            prev = self._per_cell_prev.get(cell_id)
            if prev is not None:
                r = self._reward(cell, T_s)
                if self._reward_ema is None:
                    self._reward_ema = float(r)
                else:
                    alpha = self._reward_ema_alpha
                    self._reward_ema = (1.0 - alpha) * self._reward_ema + alpha * float(r)
                self._buf.push(prev["seq"], prev["action"], r, seq_now, done_flag)
                loss = self._opt_step()
                if loss is not None:
                    self._last_loss = loss
                if self.window_len > 0:
                    self._window_reward += float(r)
                    if loss is not None:
                        self._window_loss_sum += float(loss)
                        self._window_loss_count += 1
                # Per-slice components for visibility
                (e, u, m), per_slice = self._slice_scores(cell, T_s, return_components=True)
                # Quotas snapshot
                prb_map = getattr(cell, "slice_dl_prb_quota", {}) or {}
                prb_e = float(prb_map.get(SL_E, 0))
                prb_u = float(prb_map.get(SL_U, 0))
                prb_m = float(prb_map.get(SL_M, 0))
                # Emit logs
                metrics = {
                    "reward": r,
                    "reward_ema": float(self._reward_ema) if self._reward_ema is not None else float(r),
                    "embb_score": e,
                    "urllc_score": u,
                    "mmtc_score": m,
                    "epsilon": self._epsilon() if self.train_mode else 0.0,
                    "loss": self._last_loss if self._last_loss is not None else 0.0,
                    "prb_eMBB": prb_e,
                    "prb_URLLC": prb_u,
                    "prb_mMTC": prb_m,
                    "action_prev": int(prev["action"]),
                }
                for sl, label in ((SL_E, "eMBB"), (SL_U, "URLLC"), (SL_M, "mMTC")):
                    slice_metrics = per_slice[sl]
                    for metric_name in self._slice_metrics_to_track:
                        metrics[f"slice/{label}/{metric_name}"] = float(slice_metrics.get(metric_name, 0.0))
                overall_metrics = {}
                for metric_name in self._slice_metrics_to_track:
                    vals = [float(per_slice[sl].get(metric_name, 0.0)) for sl in (SL_E, SL_U, SL_M)]
                    if metric_name in self._slice_metrics_avg:
                        denom = max(1, len(vals))
                        overall_metrics[metric_name] = sum(vals) / denom
                    else:
                        overall_metrics[metric_name] = sum(vals)
                for metric_name, value in overall_metrics.items():
                    metrics[f"slice/All/{metric_name}"] = value
                self._log(self._t, cell_id, metrics)
                if self.window_len > 0:
                    cell_sums = self._window_cell_sums.setdefault(
                        cell_id,
                        {
                            SL_E: defaultdict(float),
                            SL_U: defaultdict(float),
                            SL_M: defaultdict(float),
                            "All": defaultdict(float),
                        },
                    )
                    cell_scores = self._window_cell_score_sums.setdefault(
                        cell_id,
                        {SL_E: 0.0, SL_U: 0.0, SL_M: 0.0, "All": 0.0},
                    )
                    for sl in (SL_E, SL_U, SL_M):
                        for metric_name in self._slice_metrics_to_track:
                            cell_sums[sl][metric_name] += float(per_slice[sl].get(metric_name, 0.0))
                    cell_sums.setdefault("All", defaultdict(float))
                    for metric_name in self._slice_metrics_to_track:
                        if metric_name in self._slice_metrics_avg:
                            denom = max(1, len((SL_E, SL_U, SL_M)))
                            cell_sums["All"][metric_name] += float(overall_metrics[metric_name])  # already averaged
                        else:
                            cell_sums["All"][metric_name] += float(overall_metrics[metric_name])
                    cell_scores[SL_E] += float(e)
                    cell_scores[SL_U] += float(u)
                    cell_scores[SL_M] += float(m)
                    cell_scores.setdefault("All", 0.0)
                    cell_scores["All"] += float(e + u + m) / 3.0
                    self._window_cell_counts[cell_id] += 1

            # Choose and apply new action for next period
            a = self._act(seq_now)
            self._apply_action(cell, a)
            self._per_cell_prev[cell_id] = {"seq": seq_now.copy(), "action": a}

        # Log action histogram occasionally
        if self._tb is not None and self._t % 50 == 0:
            try:
                import numpy as np
                import torch as _torch
                counts = [self._action_counts.get(i, 0) for i in range(self._n_actions)]
                self._tb.add_histogram("actions/hist", _torch.tensor(counts, dtype=_torch.float32), self._t)
            except Exception:
                pass
        if self.window_len > 0:
            if window_terminal:
                avg_loss = (self._window_loss_sum / self._window_loss_count) if self._window_loss_count > 0 else 0.0
                if self._tb is not None:
                    try:
                        self._tb.add_scalar("episode/total_reward", float(self._window_reward), self._window_idx)
                        self._tb.add_scalar("episode/avg_loss", float(avg_loss), self._window_idx)
                        slice_iter = (
                            (SL_E, "eMBB"),
                            (SL_U, "URLLC"),
                            (SL_M, "mMTC"),
                            ("All", "All"),
                        )
                        for cell_id, slice_sums in self._window_cell_sums.items():
                            count = max(1, self._window_cell_counts.get(cell_id, 0))
                            score_sums = self._window_cell_score_sums.get(
                                cell_id,
                                {SL_E: 0.0, SL_U: 0.0, SL_M: 0.0, "All": 0.0},
                            )
                            for sl, label in slice_iter:
                                for metric_name in self._slice_metrics_to_track:
                                    mean_val = float(slice_sums.get(sl, {}).get(metric_name, 0.0)) / count
                                    self._tb.add_scalar(
                                        f"cell/{cell_id}/episode/{label}_{metric_name}_mean",
                                        mean_val,
                                        self._window_idx,
                                    )
                                score_mean = float(score_sums.get(sl, 0.0)) / count
                                self._tb.add_scalar(
                                    f"cell/{cell_id}/episode/{label}_score_mean",
                                    score_mean,
                                    self._window_idx,
                                )
                    except Exception:
                        pass
                if self._wandb is not None:
                    try:
                        import wandb
                        payload = {
                            "episode/total_reward": float(self._window_reward),
                            "episode/avg_loss": float(avg_loss),
                            "episode/length": int(self.window_len),
                        }
                        slice_iter = (
                            (SL_E, "eMBB"),
                            (SL_U, "URLLC"),
                            (SL_M, "mMTC"),
                            ("All", "All"),
                        )
                        for cell_id, slice_sums in self._window_cell_sums.items():
                            count = max(1, self._window_cell_counts.get(cell_id, 0))
                            score_sums = self._window_cell_score_sums.get(
                                cell_id,
                                {SL_E: 0.0, SL_U: 0.0, SL_M: 0.0, "All": 0.0},
                            )
                            for sl, label in slice_iter:
                                for metric_name in self._slice_metrics_to_track:
                                    mean_val = float(slice_sums.get(sl, {}).get(metric_name, 0.0)) / count
                                    payload[f"episode/cell/{cell_id}/{label}/{metric_name}_mean"] = mean_val
                                score_mean = float(score_sums.get(sl, 0.0)) / count
                                payload[f"episode/cell/{cell_id}/{label}/score_mean"] = score_mean
                        self._wandb.log(payload, step=self._window_idx)
                    except Exception:
                        pass
                self._window_idx += 1
                self._window_step = 0
                self._window_reward = 0.0
                self._window_loss_sum = 0.0
                self._window_loss_count = 0
                self._window_cell_sums.clear()
                self._window_cell_counts.clear()
                self._window_cell_score_sums.clear()
            else:
                self._window_step += 1

    def __del__(self):
        """Best‑effort cleanup of telemetry handles at interpreter shutdown."""
        try:
            self._save_model(self.model_path)
        except Exception:
            pass
        try:
            if self._tb is not None:
                self._tb.flush(); self._tb.close()
        except Exception:
            pass
        try:
            if self._wandb is not None:
                self._wandb.finish()
        except Exception:
            pass

    def to_json(self):
        """Expose a compact JSON for UI/knowledge endpoints."""
        j = super().to_json()
        j.update({
            "train_mode": self.train_mode,
            "period_steps": self.period_steps,
            "move_step": self.move_step,
            "enabled": self.enabled,
            "device": self._runtime_device,
            "log_interval": self._scalar_log_interval,
        })
        return j
