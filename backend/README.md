# AI-RAN Simulator Backend

The **AI-RAN Simulator Backend** is a Python-based simulation engine designed to model and analyze the behavior of 5G Radio Access Networks (RAN). It supports advanced features such as network slicing, mobility management, and intelligent control via xApps. This backend is part of a larger project that includes a frontend for visualization and interaction.

## 📁 Project Structure

backend/
├── main.py # Entry point for the WebSocket server
├── utils/ # Utility functions and classes
├── settings/ # Configuration files for the simulation
├── network_layer/ # network simulation logic
├── knowledge_layer/ # knowledge base, offering explanations for everything in the network layer
├── intelligence_layer/ # user-engaging and decision-making agents

---

## 📦 Requirements

- Python 3.12 or higher
- docker (to deploy the AI services)
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Start the WebSocket Server <br>Run the backend server to enable communication with the frontend:

   ```bash
   python main.py
   ```

2. Start the frontend <br>

   ```bash
   cd frontend
   npm run dev
   ```

---

## 🧪 Simple Topology (1 BS • 1 Cell • ~10 UEs)

For quick experiments and debugging, a simple preset is available. It reduces the network to one base station with a single n78 cell and spawns about 10 UEs.

Two ways to enable it:

- CLI flags (recommended)

  - Server mode (WebSocket server, controlled by the UI/client):

    ```bash
    python main.py --preset simple --ue-max 10 --mode server
    ```

  - Headless mode (no WebSocket, runs a short simulation loop and starts xApps like the KPI dashboard):

    ```bash
    python main.py --preset simple --ue-max 10 --mode headless --steps 120
    # Open the KPI dashboard at http://localhost:8061
    ```

- Environment variables (alternative)

  - Create a `.env` in `backend/` or export in shell:

    ```bash
    export RAN_TOPOLOGY_PRESET=simple
    export UE_DEFAULT_MAX_COUNT=10
    python main.py
    ```

What the preset does:

- Base stations: 1 (`bs_1`), see `settings/ran_config.py`.
- Cells: 1 n78 cell attached to `bs_1`.
- UE caps: spawn 1–2 per step, max ≈ 10 (overridable by `--ue-max` or `UE_DEFAULT_MAX_COUNT`).

Return to the full 4‑BS/8‑cell topology by omitting `--preset` (or setting `--preset default`).

### Control how many UEs per slice (simple preset)

You can explicitly choose how many UEs are subscribed to each slice when using the simple preset. These UEs attach to their single subscribed slice deterministically.

- With CLI flags:

```bash
python main.py --preset simple --ue-max 10 \
  --ue-embb 6 --ue-urllc 3 --ue-mmtc 1 
```

- With environment variables:

```bash
export RAN_TOPOLOGY_PRESET=simple
export UE_DEFAULT_MAX_COUNT=10
export UE_SIMPLE_COUNT_EMBB=6
export UE_SIMPLE_COUNT_URLLC=3
export UE_SIMPLE_COUNT_MMTC=1
python main.py
```

Notes:
- Total UEs = sum of the slice counts when `--preset simple`. If you also pass `--ue-max` and it differs, the backend adjusts the total to match the sum of slice counts.
- Omit the slice counts to keep the default randomized distribution.
- Runtime spawn is still dynamic (1–2 per step in simple mode); slice membership is fixed per IMSI.

---


---

## 🧪 Isolate PRB Effects (Freeze Mobility)

If you want KPI changes to come only from PRB allocation (and not from UE movement changing SINR/CQI/MCS), freeze mobility so UEs stay stationary.

Enable via CLI flag or environment variable (works in both server and headless modes):

```bash
# From backend/
export SIM_FREEZE_MOBILITY=1
python main.py --preset simple --mode server \
  --ue-max 3 --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --freeze-mobility

```

What this does:

- Sets all UE speeds to 0 at creation/registration and pins their targets to current positions.
- With positions fixed, radio KPIs (SINR/CQI/MCS) stay constant. DL Mbps then changes only when you adjust PRB allocation (slice shares/Move‑RB/per‑UE cap) or the offered load (traces/AI services).

Tip: You usually don’t need to freeze radio; freezing mobility is sufficient in this simulator to keep the radio constant.

---

## 📈 Replay Raw CSV Traces (per‑UE offered load)

Attach a raw packet CSV so its offered traffic is replayed and served subject to radio capacity and PRB allocation.

Downlink replay and buffering happen at the gNB (Base Station): the BS owns a per‑UE DL queue (bytes) and a per‑UE DL trace replayer (samples, clock, idx). Each simulation step, the BS advances replay clocks and enqueues due DL bytes; cells then serve from the BS queue up to capacity. The UE’s `dl_buffer_bytes` is updated as a mirror for the UI (it reflects the gNB queue).

- CLI flags:
  - `--trace-speedup <x>` (scale time; default 1.0)
  - `--strict-real-traffic` (show only served traffic; no fallback capacity)
  - `--trace-raw-map IMSI_#:path/to/raw.csv:UE_IP` (Wireshark/PCAP CSV; UE_IP required to classify DL/UL)
    You can also attach by slice or ALL UEs:
    - `--trace-raw-map slice:eMBB:path/to/embb.csv:UE_IP`
    - `--trace-raw-map ALL:path/to/trace.csv:UE_IP`
  - `--trace-bin <seconds>` (aggregation bin for raw CSV; default 1.0)
- `--trace-overhead-bytes <n>` (subtract per-packet bytes in raw CSV; default 0)
- `--trace-loop` (replay traces continuously)
- `--slice-prb SLICE=PRBs` (repeatable; sets the initial downlink PRB quota for a slice, e.g. `--slice-prb eMBB=40 --slice-prb URLLC=20`)

Key trace flags (what they do):

- `--trace-speedup <x>`: scales the replay clock. 1.0 = real time. 2.0 replays twice as fast (the same traced seconds happen in half the wall-clock time); 0.5 replays at half speed. Affects when DL samples are enqueued into the gNB DL queue; serving still happens per simulation step.
- `--trace-bin <seconds>`: aggregation window for raw packet CSVs. Packets are grouped by `floor((t - t0)/bin)*bin` and summed to produce `(t, dl_bytes, ul_bytes)` samples. Smaller bins (e.g., 0.2) preserve burstiness; larger bins (e.g., 2.0) smooth traffic. Default 1.0 aligns with the simulator’s 1 s step.
- `--trace-loop`: when enabled, traces repeat seamlessly after the last sample. Without this, each trace plays once and stops offering new bytes after the end.

Examples:

```bash
# Using raw packet CSVs (Wireshark export) — headless
python backend/main.py --preset simple --mode headless --steps 180 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Using raw packet CSVs (Wireshark export) — server
python backend/main.py --preset simple --mode server \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs (raw traces only), headless (one per slice)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility \
  --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, eMBB-only (all use eMBB trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 3 --ue-urllc 0 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, eMBB-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 3 --ue-urllc 0 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, URLLC-only (all use URLLC trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 0 --ue-urllc 3 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, URLLC-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 0 --ue-urllc 3 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, mMTC-only (all use mMTC trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 0 --ue-urllc 0 --ue-mmtc 3 \
  --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, mMTC-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 0 --ue-urllc 0 --ue-mmtc 3 \
  --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs (raw traces only), server mode (one per slice)
python backend/main.py --preset simple --mode server \
  --freeze-mobility \
  --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic


  # Three stationary UE (raw traces only), server mode (one per slice) use of mixed file seperated into 3 different files

  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/eMBB.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/URLLC.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mMTC.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE

  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/embb_gen.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/urllc_gen.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mmtc_gen.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE
  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/synthetic_embb.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/synthetic_urllc.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/synthetic_mmtc.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE with queuing 
  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/synthetic_embb_queueing.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/synthetic_urllc_queueing.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/synthetic_mmtc_queueing.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/eMBB_aligned.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/URLLC_aligned.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mMTC_aligned.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


```

How it works:
- Each UE with a trace enqueues `dl_bytes` into a per‑UE buffer at the traced times (scaled by `--trace-speedup`).
- Cells compute capacity from MCS×PRBs and serve from the buffer up to that capacity each step.
- With `--strict-real-traffic`, UE DL Mbps equals served traffic; otherwise, empty buffers show achievable capacity.

Notes:
- Traces are attached by IMSI on spawn/registration. Ensure those IMSIs exist during the run (use `--ue-max`/slice counts with simple preset).
- Place CSVs anywhere; `backend/assets/traces/` is a convenient location.
- Headless mode runs for exactly `--steps` iterations and then exits. Use server mode for an open‑ended run controlled from the frontend (http://localhost:3000) or a WebSocket client.

---

### Attach Traces to All UEs or by Slice (many UEs)

##### This part is still not working properly

When running with many UEs, you can attach traces without listing every IMSI individually:

- Same trace for all UEs (wildcard):

```bash
python backend/main.py --preset simple --mode server \
  --ue-max 30 --freeze-mobility \
  --trace-raw-map ALL:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop
```

- Different traces per slice (applies to any number of UEs in each slice):

```bash
python backend/main.py --preset simple --mode server \
  --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility \
  --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop
```

Semantics:
- `IMSI_#:` maps a specific UE.
- `ALL:` applies to any UE not matched by a specific mapping.
- `slice:<NAME>:` applies to UEs registered on that slice (`eMBB`, `URLLC`, `mMTC`).
  Priority: exact IMSI > slice mapping > ALL.

Notes:
- PRB allocation already covers all UEs in each cell; with the above, every UE will replay its trace and compete for PRBs.
- Freezing mobility keeps radio stable so differences come from offered load and PRB allocation.

### RL PRB Allocation + Traces (many UEs)

You can combine the DQN PRB allocator xApp with the multi‑UE trace mapping above to learn PRB shifts under realistic traffic. Install PyTorch first:

```bash
pip install torch
```

- Per‑slice traces with RL (any number of UEs per slice):

```bash
python backend/main.py --preset simple --mode server \
  --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility \
  --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

- Same trace for all UEs with RL:

```bash
python backend/main.py --preset simple --mode server \
  --ue-max 30 --freeze-mobility \
  --trace-raw-map ALL:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

Tips and knobs:
- `--dqn-period N` acts every N sim steps (default 1). With the default 1 s step, `N=1` roughly corresponds to T=1 s; adjust if you want a different control period.
- `--dqn-move-step K` moves K PRBs per action (Table 3 uses 1 RB).
- A pre‑trained model can be pointed to with `--dqn-model backend/models/dqn_prb.pt` (this is also the default path); omit `--dqn-train` to run inference only.
- Use the KPI dashboard with history (`--kpi-history --kpi-log`) to monitor slice PRBs, DL Mbps, buffers, and the effect of PRB moves.

## 📉 LSTM Forecast Plots (Event Histories)

Use `backend/notebooks/plot_and_predict_runner.py` to train PyTorch LSTM forecasters directly on packet event histories—no resampling. Each sample contains the last *N* packets (zero-padded when the history is shorter) so no arrivals are lost. You can choose which per-packet features to expose (e.g., `Length` only, or `Δt + Length`) or even force a uniform time grid (auto-detected minimum gap) via `--feature-sets`.

- Basic run (auto-selects CUDA if available):

  ```bash
  python backend/notebooks/plot_and_predict_runner.py backend/assets/traces/URLLC_aligned.csv \
    --epochs 15 --window 20 --output-dir backend/assets/plots
  ```

- Alternate trace and settings (force CPU, tweak batch size/hidden dim):

  ```bash
  python backend/notebooks/plot_and_predict_runner.py backend/assets/traces/eMBB_aligned.csv \
    --epochs 20 --window 30 --batch-size 16 --hidden-dim 128 --feature-sets length --device cpu
  ```
- If we want to use all three methods:
  ```bash
    python backend/notebooks/plot_and_predict_runner.py backend/assets/traces/URLLC_aligned.csv \
  --epochs 15 --window 20 --output-dir backend/assets/plots \
  --feature-sets length delta_t+length uniform-length

  python backend/notebooks/plot_and_predict_runner.py backend/assets/traces/eMBB_M3_aligned_trace.csv \
  --window 128 --hidden-dim 256 --num-layers 2 --dropout 0.2 --fc-hidden 128 \
  --include-pad-mask --use-log-target --loss huber --huber-delta 2.0 \
  --zero-weight 0.3 --val-ratio 0.1 --early-stop 10 --plateau-patience 5 --clip-grad 1.0


python backend/notebooks/prev_plot_and_predict_runner.py backend/assets/traces/eMBB_M3_aligned_trace.csv \
  --feature-sets length delta_t+length uniform-length \
  --window 128 --hidden-dim 256 --num-layers 2 \
  --val-ratio 0.1 --early-stop 10 --epochs 80 --batch-size 64

python backend/notebooks/prev_plot_and_predict_runner.py backend/assets/traces/<trace>.csv \
  --feature-sets time+length \
  --optimizer adamw --learning-rate 5e-4 \
  --loss smoothl1 --lr-scheduler plateau --lr-factor 0.6 --lr-patience 25

  ```
- Multiple traces: run once per file to populate a common folder; each call writes `traceName_epochsX_regular.png` and `traceName_epochsX_irregular.png` with axes labelled in milliseconds and bytes:

  ```bash
  for trace in backend/assets/traces/URLLC_aligned.csv backend/assets/traces/eMBB_aligned.csv; do
    python backend/notebooks/plot_and_predict_runner.py "$trace" --epochs 10 --output-dir backend/assets/plots
  done
  ```

Options:
- `--window` controls how many past packets feed the model (default 20).
- `--feature-sets` accepts one or more of `{length, delta_t+length, uniform-length}`.
  - `length`: raw event history, `Length` feature only (sequence padding handles gaps).
  - `delta_t+length`: raw event history with an extra `Δt` channel so the model learns inter-arrival spacing explicitly.
  - `uniform-length`: expands the trace onto a uniform grid using roughly the 5th percentile of observed Δt (protects against tiny jitter), zero-fills missing slots, then trains on `Length` alone. The step size is further relaxed if the grid would exceed ~2M points so training stays tractable.
- Model capacity knobs: `--hidden-dim`, `--num-layers`, `--dropout`, and `--fc-hidden` (adds an additional dense layer after the LSTM) help capture bursty traffic when the defaults underfit.
- Training behaviour knobs: choose the target space (`--target-mode raw|scaled|log`), optionally append a padding mask (`--include-pad-mask`), pick the loss (`--loss` + `--huber-delta`), rebalance zeros (`--zero-weight`), clip gradients (`--clip-grad`), and control validation (`--val-ratio`, `--early-stop`, `--plateau-patience`).
- `--device auto|cpu|cuda` to override accelerator selection (use `cpu` if GPU memory is tight).
- `--output-dir` defaults to `plots/` in the repo root if not provided.

These same padded histories drive the optional LSTM-enabled PRB allocator: set `DQN_PRB_SEQ_LEN` (>1) in `settings` to let the RL agent observe the last *N* decision states. Reduce the value or run on CPU if sequence processing becomes too heavy.


## 📊 Live KPI Dashboard xApp

Drop-in xApp that starts a Dash server at `http://localhost:8061` and streams per‑UE and per‑cell KPIs.

- Per‑UE: bitrate (Mbps), SINR, CQI, allocated PRBs.
- Per‑cell: load, PRB usage; fixed PRB quotas per slice (eMBB/URLLC/mMTC).
- Controls:
  - `Max DL PRBs per UE` cap (applies live to all cells).
  - Slice share sliders (fractions 0–1 per slice). Sum > 1 is normalized; < 1 leaves some PRBs unused.
  - “Move RB” buttons to shift PRBs between slices (paper-like actions). Default step moves 3 PRBs per click; change in `network_layer/xApps/xapp_live_kpi_dashboard.py` by editing `SLICE_MOVE_STEP_PRBS`.
  - Optional history range slider, plot history window size, and CSV logging (see below).

Slice share semantics:

- For each cell, `quota[slice] = floor(max_dl_prb_cell × share[slice])`.
- UEs in a slice share only that slice’s quota. Baseline 1 PRB/UE if possible, remainder proportional to demand.
- Per‑UE cap is enforced after slice allocation.

---

### KPI History, Range Slider, and Logging

Enable interactive history navigation on charts (range slider), configure how many points the live plots keep in memory, and optionally persist KPIs to CSV.

- CLI flags:
  - `--kpi-history`: enable a per‑chart range slider and preserve zoom/pan across live updates.
  - `--kpi-max-points <N>`: number of points kept in memory for plots (default 50). Use `0` for unbounded history.
  - `--kpi-log`: write per‑step UE/Cell KPIs to CSV files.
  - `--kpi-log-dir <path>`: output directory for KPI CSVs (default `backend/kpi_logs`).

- Environment variables (alternative):
  - `RAN_KPI_HISTORY_ENABLE=1`
  - `RAN_KPI_MAX_POINTS=<N>` (0 = unbounded)
  - `RAN_KPI_LOG_ENABLE=1`
  - `RAN_KPI_LOG_DIR=<path>`

- Example (server mode):
```bash
python backend/main.py --preset simple --mode server \
  --kpi-history --kpi-max-points 10000 --kpi-log --kpi-log-dir backend/kpi_logs
# KPI dashboard at http://localhost:8061
```

Notes:
- With the history slider enabled, legends are placed at the top to avoid overlap with the slider.
- Unbounded history (0) grows with runtime and number of UEs; prefer a large but finite window for long runs (e.g., 5000–20000).
- CSV logs include one row per UE and per cell per step. UE CSV columns: `sim_step, imsi, dl_bps, dl_mbps, sinr_db, cqi, dl_buffer_bytes, dl_prb_granted, dl_prb_requested, dl_latency_ms`. Cell CSV: `sim_step, cell_id, dl_load, allocated_prb, max_prb`.

---

## 🧠 Example xApps

Example xApps are located in the `network_layer/xApps/` directory:

- Blind Handover xApp: Implements handover decisions based on RRC Event A3.
- AI service monitoring xApp: Monitors the AI service performance and provides insights.
 - Live KPI Dashboard xApp: Real‑time UE/Cell KPIs with per‑UE cap and per‑slice PRB controls.
- DQN PRB Allocator xApp: Learns to shift DL PRBs among eMBB/URLLC/mMTC slices using a DQN policy inspired by the Tractor paper.
- Episodic DQN PRB Allocator xApp: Automates offline RL by replaying a catalog of UE/trace scenarios one episode at a time, resetting the simulator between traces so the DQN trains on finite windows instead of the continual stream.
- Gym PRB Allocator xApp: Clean-room, two-slice (eMBB/URLLC) DQN agent that speaks a Gym-style API (reset/step) and runs over fully scripted episodes defined in JSON.
- SB3 DQN PRB Allocator xApp: Same objective as the DQN PRB allocator but powered by Stable-Baselines3.

To load custom xApps, add them to the xApps/ directory and ensure they inherit from the xAppBase class.

### DQN PRB Allocator xApp

The DQN xApp implements a small DQN agent that observes per‑cell state and applies the 7 actions from Table 3 in the Tractor paper (move one PRB between slices or keep).

- State per cell: normalized UE mix, per-slice PRB share, served throughput, buffer backlog, PRB demand, and grant satisfaction ratios for mMTC/URLLC/eMBB (18 features).
- Actions: each slice (eMBB/URLLC/mMTC) chooses `{-1, 0, +1}` to decrease keep or increase its quota by `DQN_PRB_MOVE_STEP` PRBs, yielding `3^3 = 27` discrete actions (all-zero means keep current quotas).
- Reward: weighted sum of per-slice scores that scale with slice demand (eMBB: throughput & drained queues under high pressure, URLLC: exponential delay penalty & reliability, mMTC: efficient use of reserved PRBs while avoiding oversupply), each normalised to [0,1].
- Fine-tune how aggressively demand boosts each slice score via `DQN_NEED_SATURATION` (default `1.5`, higher = more headroom before the need term saturates).

- State normalization: `backend/network_layer/xApps/xapp_dqn_prb_allocator.py:270-320` scales UE counts by `UE_DEFAULT_MAX_COUNT`, PRB quotas by each cell’s `max_dl_prb`, throughput by `DQN_NORM_MAX_DL_MBPS`, buffers by `DQN_NORM_MAX_BUF_BYTES`, and demand/grant ratios to keep all features near `[0,1]`.
- Network/optimizer: both the handcrafted DQN xApp and the SB3 variant use a fully-connected network `18 → 256 → 7` with ReLU and the Adam optimiser at learning rate `0.01`, matching the Tractor paper’s configuration while automatically adjusting to our state/action dimensions.

Enable it at runtime (requires `torch`):

```bash
pip install torch  # if not already installed

# Server mode with DQN (online training), the KPI dashboard, and range slider
python backend/main.py --preset simple --mode server \
  --dqn-prb --dqn-train --dqn-period 2 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

Useful flags/env vars:
- `--dqn-prb` (or `DQN_PRB_ENABLE=1`): enable the DQN xApp.
- `--dqn-train` (or `DQN_PRB_TRAIN=1`): enable online training. Omit it to run in pure evaluation/inference mode (weights stay frozen).
- `--dqn-model <path>` (or `DQN_PRB_MODEL_PATH`): save/load model weights (default `backend/models/dqn_prb.pt`).
- `--dqn-period <steps>` (`DQN_PRB_DECISION_PERIOD_STEPS`): act every N sim steps (default 2 → ~100 ms with the 50 ms sim step).
- `--dqn-move-step <PRBs>` (`DQN_PRB_MOVE_STEP`): PRBs moved per action (default 1).
- `--dqn-lr <value>` (`DQN_PRB_LR`): learning rate for the custom DQN (default 1e-2).
- `--dqn-gamma <value>` (`DQN_PRB_GAMMA`): discount factor (default 0.99).
- `--dqn-target-update <steps>` (`DQN_PRB_TARGET_UPDATE`): decision steps between target-network syncs (default 200).
- `--dqn-save-interval <steps>` (`DQN_PRB_SAVE_INTERVAL`): checkpoint the custom DQN every N decisions (writes both the base file and a `_stepN` snapshot); 0 disables periodic saves.
- `--dqn-device <device>` (`DQN_PRB_DEVICE`): execution device for the custom DQN (`auto`, `cpu`, `cuda`, `cuda:0`, …); defaults to auto-select GPU when available.
- `--dqn-log-interval <steps>` (`DQN_PRB_LOG_INTERVAL`): log TensorBoard/W&B scalars every N decisions instead of every step (default 1).
- `--dqn-aux-future-state` (`DQN_PRB_AUX_FUTURE_STATE`): add an auxiliary head that predicts the next state vector to shape temporal representations.
- `--dqn-aux-weight <λ>` (`DQN_PRB_AUX_WEIGHT`): loss weight for the auxiliary prediction head (default 0.1).
- `--dqn-aux-horizon <steps>` (`DQN_PRB_AUX_HORIZON`): number of future steps used as the auxiliary prediction target (default 1).
- `--dqn-episode-len <steps>` (`DQN_PRB_EPISODE_LEN`): treat every N decisions as a pseudo-episode for logging/bootstrapping (0 disables; default 0).
- `--dqn-episode-no-done` (or `DQN_PRB_EPISODE_MARK_DONE=0`): keep transitions non-terminal at pseudo-episode boundaries.
- `DQN_PRB_DEBUG_STATE=1` (and optional `DQN_PRB_DEBUG_INTERVAL=N`): log raw slice KPIs plus the normalised 18-value observation vector every N decisions for easier inspection.
- `DQN_PRB_MODEL_ARCH` / `--dqn-model-arch`: choose the feature extractor for the custom DQN (`mlp`, `lstm`, `tcn`, `seq2seq`). Non-MLP options consume the last `DQN_PRB_SEQ_LEN` (or `--dqn-seq-len`) state snapshots (>=2 recommended).
- `DQN_PRB_SEQ_LEN` / `--dqn-seq-len`: number of consecutive observations fed to sequential extractors (minimum enforced at 2).
- `DQN_PRB_SEQ_HIDDEN` / `--dqn-seq-hidden`: hidden size for sequential extractors (default 128).
- `SIM_STEP_TIME_DEFAULT=<seconds>`: set the simulated wall-clock duration of each step (default 1.0 s). Combined with `--dqn-period` this controls how often the PRB allocator acts.
- Exploration and learning hyper‑params can be set via env vars: `DQN_PRB_EPSILON_START`, `DQN_PRB_EPSILON_END`, `DQN_PRB_EPSILON_DECAY`, `DQN_PRB_LR`, `DQN_PRB_BATCH`, `DQN_PRB_BUFFER`, `DQN_PRB_GAMMA`.
- `--ws-host` / `WS_SERVER_HOST`: WebSocket host for the backend (default `localhost`).
- `--ws-port` / `WS_SERVER_PORT`: WebSocket port for the backend (default `8760`).
- `--dash-port` / `DASH_PORT`: port for the live KPI dashboard xApp (default `8050`).
- Frontend instances read `NEXT_PUBLIC_WS_HOST`, `NEXT_PUBLIC_WS_PORT`, and `NEXT_PUBLIC_WS_PROTOCOL` to decide which backend WebSocket to connect to.

#### Episodic training mode

Need finite-length rollouts instead of the continual stream? Enable the episodic variant of the xApp:

```bash
python backend/main.py --preset simple --mode headless \
  --dqn-prb-episodic --dqn-train \
  --dqn-episode-config backend/assets/episodes/sample_config.json
```

New CLI/env switches:

- `--dqn-prb-episodic` (`DQN_PRB_EPISODIC_ENABLE=1`): activate the episodic xApp (the standard DQN xApp stays off).
- `--dqn-episode-config <path>` (`DQN_EPISODE_CONFIG_PATH`): JSON file describing the episode catalog.
- `--dqn-episode-config-json '<json>'` (`DQN_EPISODE_CONFIG_JSON`): inline JSON alternative.
- `--dqn-episode-loop` (`DQN_EPISODE_LOOP=1`): restart from the first scenario after consuming the list.

Each entry in `backend/assets/episodes/sample_config.json` demonstrates the schema:

- `id`: friendly label for logs.
- `duration_steps`: number of DQN *decisions* (not simulator ticks) to run before resetting.
- `slice_prb`: optional per-slice PRB quotas applied to every cell at episode start.
- `ue_prb_cap`: optional per-UE DL cap.
- `freeze_mobility`: keep spawned UEs stationary (handy for reproducible traces).
- `ue_groups`: list of cohorts with `slice`, `count`, and optional traffic descriptors (`trace`/`trace_file`, `ue_ip`, `trace_speedup`, `trace_bin`).

At the start of each episode the xApp clears any auto-spawned UEs, spawns the requested groups, attaches their traces directly at the gNB, resets the replay buffer queues, and runs for `duration_steps` decisions. The last transition in the episode is flagged with `done=True` so the replay buffer learns from terminal rollouts. When an episode finishes the simulator is reset automatically and the next scenario loads—no dashboard/front-end required for training loops.

To point the React frontend at a specific backend instance, set these environment variables when launching `npm run dev` (or create a `frontend/.env.local`):

```bash
cd frontend
NEXT_PUBLIC_WS_HOST=localhost \
NEXT_PUBLIC_WS_PORT=8760 \

export NEXT_PUBLIC_WS_HOST=localhost
export NEXT_PUBLIC_WS_PORT=8760
npm run dev
```

On Windows PowerShell:

```powershell
cd frontend
$env:NEXT_PUBLIC_WS_HOST = "localhost"
$env:NEXT_PUBLIC_WS_PORT = "8780"
npm run dev
```

Any values you export become part of the generated WebSocket URL (`ws://<host>:<port>`), enabling multiple backend/frontend pairs to run side-by-side.

Notes:
- The xApp applies actions after each simulator step; changes take effect in the next allocation round.
- Rewards use current KPIs and are a practical instantiation of the paper’s formulas; you can refine the shaping or weights in `settings/ran_config.py`.
- If `torch` is unavailable, the xApp disables itself gracefully.
- TensorBoard logging includes both the per-step reward and an exponential moving average (`reward_ema`) so you can quickly spot convergence trends.

#### Sequential feature extractors (custom DQN)

The handcrafted DQN xApp can swap its feature extractor between the default MLP and the sequence-aware models defined in `seq_models.py`. When `DQN_PRB_MODEL_ARCH` (or `--dqn-model-arch`) is set to `lstm`, `tcn`, or `seq2seq`, the xApp keeps a deque of the last `DQN_PRB_SEQ_LEN` states per cell (minimum enforced at 2) and feeds a `[batch, seq_len, state_dim]` tensor into the selected module, which outputs the 7 Q-values directly. Examples:

```bash
# LSTM encoder over the last 8 observations
python backend/main.py --preset simple --mode server \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --dqn-model-arch lstm --dqn-seq-len 8

# Temporal convolutional network (TCN)
python backend/main.py --preset simple --mode server \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --dqn-model-arch tcn --dqn-seq-len 8

# Seq2Seq attention extractor
python backend/main.py --preset simple --mode server \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --dqn-model-arch seq2seq --dqn-seq-len 8


 python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces_long/eMBB_M4_aligned_trace_long.csv:10.0.0.4 --trace-raw-map IMSI_1:backend/assets/traces_long/URLLC_M1_aligned_trace_long.csv:10.0.0.1 --trace-raw-map IMSI_2:backend/assets/traces_long/mMTC_M5_aligned_trace_long.csv:10.0.0.5 --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --sim-step 1 --dqn-model-arch lstm --dqn-seq-hidden 8

  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces/eMBB_constant.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/traces/URLLC_constant.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/traces/mMTC_constant.csv:172.30.1.250 --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --sim-step 1 --dqn-model-arch lstm --dqn-seq-hidden 8

  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces/eMBB_constant.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/traces/URLLC_constant.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/traces/mMTC_constant.csv:172.30.1.250 --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-episode-len 1000 --dqn-episode-no-done --dqn-save-interval 10000 

  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces/eMBB_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/traces/URLLC_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/traces/mMTC_constant_sec.csv:172.30.1.250 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-episode-len 1000 --dqn-episode-no-done --dqn-save-interval 10000 --trace-bin 0.001 --sim-step 0.05



  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces/eMBB_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/traces/URLLC_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/traces/mMTC_constant_sec.csv:172.30.1.250 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --trace-bin 0.05 --sim-step 0.05


  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/traces/eMBB_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/traces/URLLC_constant_sec.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/traces/mMTC_constant_sec.csv:172.30.1.250 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop  --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --trace-bin 0.05 --sim-step 0.05 -trace-debug
###########this structure of the episode does not work properly


  python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


  --dqn-device cuda
  --dqn-log-interval 100

###################### for evaluation #########################
    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-model backend/models/past/dqn_prb_mlp_step100000.pt --dqn-prb  --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5  --dqn-save-interval 10000 --trace-bin 0.05 --sim-step 0.05 --dqn-epsilon-start 0 --dqn-epsilon-end 0


    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-model backend/models/dqn_prb_lstm_seq2_step100000.pt --dqn-prb  --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 --dqn-model-arch lstm --dqn-seq-hidden 8  --trace-bin 0.05 --sim-step 0.05 --dqn-epsilon-start 0 --dqn-epsilon-end 0  

python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 \
  --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 \
  --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 \
  --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
  --trace-bin 0.05 --sim-step 0.05 \
  --dqn-prb --dqn-model backend/models/dqn_prb_lstm_seq2_20251015_225859_step100000.pt\
  --dqn-period 1 --dqn-move-step 5 \
  --dqn-model-arch lstm --dqn-seq-len 8 --dqn-seq-hidden 8 \
  --dqn-aux-future-state --dqn-aux-horizon 4 --dqn-aux-weight 0.1 \
  --dqn-epsilon-start 0 --dqn-epsilon-end 0 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 







python -m backend.tools.convert_training_configs_to_gym_catalog \
  --input backend/notebooks/xapp_dqn_training_configs.json \
  --trace-root backend/notebooks/Unified_CMTC/traces/aligned \
  --output backend/assets/episodes/gym_from_training_config.json \
  --sim-step 0.002 \
  --decision-period 1


  python -m backend.tools.convert_training_configs_to_gym_catalog \
  --input backend/notebooks/xapp_dqn_training_configs.json \
  --trace-root backend/notebooks/Unified_CMTC/traces/aligned \
  --output backend/assets/episodes/gym_from_training_config.json \
  --sim-step 0.002 --decision-period 1 \
  --embb-ue-ip 10.0.0.2 --urllc-ue-ip 10.0.0.1





##############################################################
    python backend/main.py --preset simple --mode server \
    --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
    --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 \
    --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 \
    --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 \
    --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
    --trace-bin 0.05 --sim-step 0.05 \
    --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 \
    --dqn-model-arch lstm --dqn-seq-len 8 --dqn-seq-hidden 8 \
    --dqn-aux-future-state --dqn-aux-horizon 4 --dqn-aux-weight 0.1 \
    --dqn-save-interval 10000 --dqn-log-tb -slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5








      python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --trace-bin 0.05 --sim-step 0.05 -trace-debug


      python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5  --dqn-save-interval 10000 --trace-bin 0.01 --sim-step 0.01 --dqn-log-interval 100 --dqn-device cuda

    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5  --dqn-save-interval 10000 --trace-bin 0.05 --sim-step 0.05


    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 --dqn-model-arch lstm --dqn-seq-hidden 8  --trace-bin 0.05 --sim-step 0.05

    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 --dqn-model-arch tcn --dqn-seq-hidden 8  --trace-bin 0.05 --sim-step 0.05

    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 --dqn-model-arch seq2seq --dqn-seq-hidden 8  --trace-bin 0.05 --sim-step 0.05




# timestep episode does not work correctly
python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 --dqn-model-arch seq2seq --dqn-seq-hidden 8

python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/URLLC_M1_aligned_trace_sec.csv:10.0.0.1 --trace-raw-map IMSI_1:backend/assets/New_traces/eMBB_M2_aligned_trace_sec.csv:10.0.0.2 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace_sec.csv:10.0.0.3 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-save-interval 10000 

    # With our new data
    python backend/main.py --preset simple --mode server --freeze-mobility --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 --trace-raw-map IMSI_0:backend/assets/New_traces/eMBB_M2_aligned_trace.csv:172.30.1.250 --trace-raw-map IMSI_1:backend/assets/New_traces/URLLC_M1_aligned_trace.csv:172.30.1.250 --trace-raw-map IMSI_2:backend/assets/New_traces/mMTC_M3_aligned_trace.csv:172.30.1.250 --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 5 --kpi-history --kpi-log --dqn-log-tb --slice-prb eMBB=5 --slice-prb URLLC=5 --slice-prb mMTC=5 --dqn-episode-len 1000 --dqn-episode-no-done --dqn-save-interval 10000 --sim-step 0.05
```

You can achieve the same configuration via environment variables if you prefer, e.g. `DQN_PRB_MODEL_ARCH=seq2seq DQN_PRB_SEQ_LEN=8`. Set `--dqn-seq-hidden <dim>` (or `DQN_PRB_SEQ_HIDDEN`) to adjust the hidden width of the sequential extractor (default 128).

Omit `--dqn-model-arch` (or set it to `mlp`) to keep the original feed-forward policy. The SB3-backed PRB allocator always uses MLP features today; if you want LSTM/TCN/Seq2Seq, stick with the custom DQN path above.

Example: to evaluate a near-RT loop that reallocates PRBs roughly every 50 ms, run the simulator with a 50 ms step and keep the decision period at one tick:

```bash
python backend/main.py --preset simple --mode server \
  --sim-step 0.05 --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

With the default (1 s) step the same command would act every second; reducing the step pushes the DQN to react at 20 Hz, which sits comfortably inside the 10–100 ms near-RT budget typically cited for xApps that steer slice resources.

**Timing quick reference**
- Raw trace events preserve their original ordering but are scaled by `TRACE_SPEEDUP` (default 1.0). Values >1 speed the traffic up; <1 slow it down.
- The simulator advances by `SIM_STEP_TIME_DEFAULT` every loop (default 1.0 s). During each step it injects all trace traffic scheduled for that interval, updates UE buffers, and runs every enabled xApp.
- The PRB DQN only acts when the loop index is divisible by `DQN_PRB_DECISION_PERIOD_STEPS`. Effective decision cadence = `SIM_STEP_TIME_DEFAULT × DQN_PRB_DECISION_PERIOD_STEPS`.
  * Default settings (`SIM_STEP_TIME_DEFAULT=1.0`, `dqn-period=1`): one PRB reallocation per second.
  * Near-RT example above (`SIM_STEP_TIME_DEFAULT=0.05`, `dqn-period=1`): one reallocation every 50 ms.
  * `SIM_STEP_TIME_DEFAULT=0.01` with `dqn-period=5`: still 50 ms cadence, but the simulator runs 10× more iterations per second.
- `DQN_PRB_MOVE_STEP` controls how many PRBs shift per action; it does not change the cadence.
- TensorBoard log directories, default W&B run names, and the default checkpoint path automatically append the active architecture (e.g. `dqn_prb_lstm_seq8_<timestamp>` / `backend/models/dqn_prb_lstm_seq8.pt`), making it easier to compare configurations.
- The simulation loop stops automatically after `SIM_MAX_STEP` iterations (default 2 000 000). Bump the `SIM_MAX_STEP` environment variable if you need longer runs.

### SB3 DQN PRB Allocator xApp

`xapp_sb3_dqn_prb_allocator.py` mirrors the state/action/reward design of the custom DQN but uses the Stable-Baselines3
implementation under the hood so you can validate results or reuse SB3 tooling. Enable it with:

```bash
pip install stable-baselines3 gymnasium  # if not already installed

python backend/main.py --preset simple --mode server \
  --sb3-dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1
```

Key toggles/env vars:
- `--sb3-dqn-prb` (or `SB3_DQN_PRB_ENABLE=1`): enable the SB3-backed xApp.
- `SB3_DQN_MODEL_PATH`: checkpoint path (defaults to `backend/models/dqn_prb_sb3.zip`).
- `SB3_DQN_TOTAL_STEPS`, `SB3_DQN_TARGET_UPDATE`, `SB3_DQN_SAVE_INTERVAL`: tune learning horizon, target-network refresh, and autosave cadence.

The xApp falls back to inference-only if SB3/Gymnasium are unavailable.

Logging mirrors the custom DQN xApp, so you can reuse the same flags:

- TensorBoard: `python backend/main.py ... --sb3-dqn-prb --dqn-train --dqn-log-tb --dqn-tb-dir backend/tb_logs`
- Weights & Biases: `python backend/main.py ... --sb3-dqn-prb --dqn-train --dqn-wandb --dqn-wandb-project ai-ran-dqn`

Both commands share the rest of the arguments with your run (UE counts, traces, etc.). Internally they set `DQN_TB_ENABLE` / `DQN_WANDB_ENABLE`, which the SB3 allocator now honors alongside the custom DQN.

#### DQN Training Telemetry (TensorBoard / W&B)

You can visualize training with TensorBoard or Weights & Biases (optional):

- TensorBoard (recommended locally):
  - Install: `pip install tensorboard`
  - Run with logging:
    ```bash
    python backend/main.py --preset simple --mode server \
      --dqn-prb --dqn-train --dqn-log-tb --dqn-tb-dir backend/tb_logs \
      --kpi-history
    ```
  - Launch TensorBoard: `tensorboard --logdir backend/tb_logs`
  - You’ll see per‑cell reward, slice scores (eMBB/URLLC/mMTC), loss, epsilon, PRB quotas, and action histograms.

- Weights & Biases (cloud):
  - Install and login: `pip install wandb && wandb login`
  - Run with logging:
    ```bash
    python backend/main.py --preset simple --mode server \
      --dqn-prb --dqn-train --dqn-wandb \
      --dqn-wandb-project ai-ran-dqn --dqn-wandb-name local-run-1
    ```
  - The same metrics are logged to your W&B project.

- Sample code run in tensorboard:
```
    python backend/main.py --preset simple --mode server   --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility   --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1   --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1   --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1   --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop   --dqn-prb --dqn-train --dqn-period 2 --dqn-move-step 1   --kpi-history --kpi-log --dqn-log-tb --dqn-tb-dir backend/tb_logs
```
---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests to improve the simulator.

---

## 📬 Contact

For questions or support, please feel free to open issues.
