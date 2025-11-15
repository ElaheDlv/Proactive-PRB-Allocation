# AI-RAN Simulator Backend

This backend is a customized build of the AI-RAN Simulator that we use to study proactive PRB allocation policies, replay measured traffic, and evaluate xApps under controlled topologies. It exposes a WebSocket API for the UI, supports a headless loop for offline RL training, and ships with tooling to create repeatable trace catalogs.

---

## 📁 Key Directories

- `main.py` – entry point that wires CLI flags/environment variables and starts the WebSocket server or headless loop.
- `settings/` – tunables grouped by domain (`sim_config.py`, `ran_config.py`, `slice_config.py`, etc.). Every value can be overridden via environment variables or CLI flags exposed by `main.py`.
- `network_layer/` – radio/link simulation, UE/base-station models, schedulers, and xApps (constant allocator, DQN agents, KPI dashboard, etc.).
- `intelligence_layer/` & `knowledge_layer/` – RIC logic, explainability helpers, and orchestration utilities.
- `utils/` – shared helpers, parsers, trace loaders.
- `tools/` – standalone scripts (episode catalog converter, trace utilities).
- `notebooks/` – traffic generation notebooks (Unified CMTC, CMDP generators, plotting helpers) plus JSON configs used by the xApps.
- `assets/` – aligned trace CSVs, episodic catalogs, and other binary artifacts consumed at runtime.
- `tb_logs/`, `evaluations/`, `models/` – training logs, evaluation runs, and stored checkpoints.

---

## ✅ Requirements & Installation

- Python **3.12+**
- Optional CUDA toolkit if training torch-based agents on GPU
- Node.js (only if you plan to run the frontend)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Tip: copy `backend/.env.example` (if present) to `.env` to persist frequently used overrides such as topology presets or logging paths.

---

## 🚀 Running the Backend

The backend runs in two modes:

| Mode    | Description |
|---------|-------------|
| `server` (default) | Spins up the WebSocket server (default `ws://localhost:8763`) plus the Dash KPI dashboard (default `http://localhost:8059`). Use this when driving the UI. |
| `headless` | Runs the simulator loop without any sockets. Useful for RL training, sweeps, or scripted evaluations. Requires `--steps` to bound the run. |

Launch with:

```bash
python main.py [--mode server|headless] [options...]
```

### Topology & UE Controls

- `--preset default|simple` – the simple preset keeps one gNB/n78 cell active.
- `--ue-max N` – cap simultaneous UEs (overrides `UE_DEFAULT_MAX_COUNT`).
- `--ue-embb/--ue-urllc/--ue-mmtc` – in the simple preset, deterministically spawn exactly this many subscribers per slice.
- `--freeze-mobility` (or `SIM_FREEZE_MOBILITY=1`) – pin UE locations to keep SINR/MCS constant so PRB changes are isolated.
- `--sim-step S` – simulation step length in seconds (default 1.0).
- `--steps K` – number of steps to run in headless mode.

Example (server mode, deterministic slice mix, stationary UEs):

```bash
python main.py --preset simple --mode server \
  --ue-embb 4 --ue-urllc 3 --ue-mmtc 3 \
  --freeze-mobility --sim-step 0.5
```

Example (headless run with a capped UE population):

```bash
python main.py --preset simple --mode headless \
  --steps 1200 --ue-max 12 --freeze-mobility
```

---

## 📡 Traffic & Mobility

### Freeze Mobility

`--freeze-mobility` or `SIM_FREEZE_MOBILITY=1` sets UE speeds to zero and locks the target position upon registration. This is the recommended switch when comparing PRB policies because radio quality then becomes a pure function of the configured traces and PRB allocations.

### Real Trace Replay

Attach per-UE or per-slice CSV traces via `--trace-raw-map`. Each mapping takes one of the following forms:

- `IMSI_#:path/to/trace.csv:UE_IP`
- `slice:eMBB:path/to/trace.csv:UE_IP`
- `ALL:path/to/trace.csv:UE_IP`

Important flags:

- `--trace-speedup X` – compress/expand the trace timeline.
- `--trace-bin seconds` – aggregate packets before enqueuing. Use `0` (or negative) to replay the original timestamps.
- `--trace-loop` – restart the trace when it reaches the end.
- `--trace-overhead-bytes N` – subtract headers/trailers before the bytes are added to the buffers.
- `--strict-real-traffic` – report only the replayed traffic (disables fallback throughput smoothing).
- `--trace-debug[(-imsi)]` – inspect replay buffers.
- `--trace-validate-only` – ensure files exist and exit.

Example (one UE per slice, looping traces, strict replay):

```bash
python main.py --preset simple --mode server --freeze-mobility \
  --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/traces/eMBB_aligned.csv:10.0.0.2 \
  --trace-raw-map IMSI_1:backend/assets/traces/URLLC_aligned.csv:10.0.0.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mMTC_aligned.csv:10.0.0.3 \
  --trace-bin 1 --trace-speedup 1 --trace-loop --strict-real-traffic
```

---

## 🧠 xApps & Control Loops

All xApps live in `network_layer/xApps/`. Enable them through CLI flags (which set the corresponding env vars) or directly via the settings files.

### Live KPI Dashboard (`xapp_live_kpi_dashboard.py`)

- Starts automatically when `main.py` is run in server mode.
- Use `--dash-port` to change the HTTP port.
- `--kpi-history`, `--kpi-max-points N`, `--kpi-log`, `--kpi-log-dir path` control caching and CSV logging.
- Interactive controls include per-slice PRB sliders, Move-RB buttons, and per-UE PRB caps.

### Constant PRB Allocator (`xapp_prb_constant_allocator.py`)

Baseline allocator that pins slice quotas to known values and logs average throughput/latency.

```bash
python main.py --prb-const --prb-const-embb 40 --prb-const-urllc 20 \
  --prb-const-log-interval 2000
```

### Gym-style PRB Allocator (`xapp_prb_gym_allocator.py`)

A clean-room environment that exposes a Gym-like `reset/step` API and maintains its own DQN (MLP or LSTM) per the episode catalog stored in `PRB_GYM_CONFIG_PATH`.

Key flags:

- `--prb-gym --prb-gym-config backend/assets/episodes/<file>.json`
- `--prb-gym-loop`, `--prb-gym-shuffle`
- `--prb-gym-eps-decay N` – linear epsilon decay across total decisions
- `--prb-gym-load-model path` – resume training/eval
- `--prb-gym-eval` – skip replay buffer updates for pure evaluation
 - `--dqn-model path/to/model.pt` – load/save checkpoints for the Gym agent.
 - DQN hyperparameters stay under the familiar `--dqn-*` switches:
   - `--dqn-period`, `--dqn-move-step`, and `--dqn-max-train-steps` to control action cadence.
   - `--dqn-epsilon-start/end/decay`, `--dqn-lr`, `--dqn-gamma`, `--dqn-target-update`, `--dqn-save-interval`, `--dqn-log-interval`.
   - `--dqn-model-arch mlp|lstm|tcn|seq2seq`, `--dqn-seq-len`, `--dqn-seq-hidden`, `--dqn-device`.
   - `--dqn-log-tb`, `--dqn-tb-dir` to emit TensorBoard traces.

### Other xApps

`xapp_A3_handover_blind.py` (A3-based HO), `xapp_AI_service_monitor.py` (health monitoring), and helper modules remain in the tree. Enable them from `settings/ric_config.py` when needed.

---

## 📊 Trace Catalogs & Tooling

### Unified CMTC Traffic Generator

The notebooks under `notebooks/Unified_CMTC*/` create aligned traces from CTMC/traffic parameters. Typical workflow:

1. Configure the scenario in `config_for_run.ipynb` or `Unified_Final_CMTC_traffic_generator*.ipynb`.
2. Export aligned per-slice CSVs to `backend/notebooks/Unified_CMTC/.../traces/aligned/`.
3. Copy or symlink the generated traces into `backend/assets/traces/` (or reference directly via `--trace-raw-map`).

### Plotting & Forecasting

`backend/notebooks/plot_and_predict_runner.py` trains PyTorch LSTMs over packet histories and generates plots for trace sanity checks. Use it to visualise new traces before feeding them to the simulator:

```bash
python backend/notebooks/plot_and_predict_runner.py backend/assets/traces/eMBB_aligned.csv \
  --epochs 20 --window 30 --output-dir backend/assets/plots
```

### Convert Training Configs to Gym Episodes

`backend/tools/convert_training_configs_to_gym_catalog.py` translates the legacy `xapp_dqn_training_configs*.json` files produced by the notebooks into the episodic format consumed by the Gym xApp.

```bash
python -m backend.tools.convert_training_configs_to_gym_catalog \
  --input backend/notebooks/xapp_dqn_training_configs.json \
  --trace-root backend/notebooks/Unified_CMTC/Trace_20s/traces/aligned \
  --output backend/assets/episodes/gym_trace_20s.json \
  --sim-step 0.05 --decision-period 1 --trace-speedup 1.0 --trace-bin 0 \
  --embb-ue-ip 10.0.0.2 --urllc-ue-ip 10.0.0.1
```

---

## ⚙️ Configuration Cheat Sheet

- `settings/sim_config.py` – step length, mobility toggles, trace replay defaults.
- `settings/ran_config.py` – topology presets, base-station definitions, slice names, constant PRB map.
- `settings/slice_config.py` – slice KPIs, PRB budgets, UE load targets.
- `settings/ue_config.py` – UE radio profiles, mobility models.
- `settings/ric_config.py` – xApp enable flags and controller wiring.
- `settings/agent_config.py` – RL hyperparameters (gamma, epsilon, replay buffer sizes, logging paths).
- `settings/ws_server_config.py` – WebSocket host/port.

Every setting can be overridden at runtime. Example:

```bash
SIM_STEP_TIME_DEFAULT=0.05 TRACE_BIN=0 TRACE_LOOP=1 python main.py --mode headless
```

---

## 🧾 Logs & Artifacts

- `backend/kpi_logs/` – created when `--kpi-log` is enabled (per-step UE/cell CSVs).
- `tb_logs/` – TensorBoard summaries from `--dqn-log-tb` or the Gym xApp.
- `models/` – saved DQN checkpoints; default path is `backend/models/dqn_prb.pt`.
- `assets/episodes/` – episodic catalogs for the Gym style or episodic DQN modes.
- `evaluations/` – scripted evaluation configs (see `evaluations/IEEE_comm_mag_25/` for reference).

---

## 🧪 Typical Workflow

1. **Generate traces** using the Unified CMTC notebook suite and export aligned CSVs.
2. **Create training configs** (`xapp_dqn_training_configs_*.json`) with the same notebook.
3. **Convert** configs to a Gym catalog when working with episodic/Gym runs.
4. **Train** via `python main.py --mode headless --preset simple --freeze-mobility \ ... --dqn-prb --dqn-train` or `--prb-gym`.
5. **Evaluate** the resulting checkpoint in server mode together with the frontend/KPI dashboard.

This README now documents only the components that are actively maintained in the current workflow. If you re-enable any legacy behaviour (e.g., other xApps or mobility models), add the relevant notes here so the team keeps a single source of truth.
