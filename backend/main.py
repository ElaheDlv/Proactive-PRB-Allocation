import os
import argparse
import asyncio
import json
from functools import partial

import importlib
import dotenv

# -------------------------------------------------------------
# CLI args to control topology preset and run mode before import
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="AI-RAN Simulator Backend")
parser.add_argument("--preset", choices=["default", "simple"], help="Topology preset")
parser.add_argument("--ue-max", type=int, help="Override UE_DEFAULT_MAX_COUNT")
parser.add_argument("--ue-embb", type=int, help="Simple preset: number of UEs with eMBB subscription")
parser.add_argument("--ue-urllc", type=int, help="Simple preset: number of UEs with URLLC subscription")
parser.add_argument("--ue-mmtc", type=int, help="Simple preset: number of UEs with mMTC subscription")
parser.add_argument(
    "--freeze-mobility",
    action="store_true",
    help="Freeze UE mobility (speed=0; pin targets to current position).",
)
parser.add_argument(
    "--mode", choices=["server", "headless"], default="server", help="Run as WebSocket server or headless loop",
)
parser.add_argument("--ws-host", type=str, help="WebSocket server host (default localhost)")
parser.add_argument("--ws-port", type=int, help="WebSocket server port (default 8763)")
parser.add_argument("--dash-port", type=int, help="Port for the live KPI dashboard (default 8059)")
parser.add_argument("--steps", type=int, default=120, help="Headless: number of steps to run")
parser.add_argument("--sim-step", type=float, help="Simulation step duration in seconds (default 1.0)")
# KPI dashboard controls
parser.add_argument(
    "--kpi-max-points",
    type=int,
    help="Live KPI plot history length (points); 0 = unbounded",
)
parser.add_argument(
    "--kpi-history",
    action="store_true",
    help="Enable interactive history slider on KPI charts",
)
parser.add_argument(
    "--kpi-log",
    action="store_true",
    help="Enable CSV logging of per-step KPIs",
)
parser.add_argument(
    "--kpi-log-dir",
    type=str,
    help="Directory to write KPI CSV logs",
)

# DQN PRB allocator controls
parser.add_argument("--dqn-prb", action="store_true", help="Enable DQN PRB allocator xApp")
parser.add_argument(
    "--dqn-prb-simple",
    action="store_true",
    help="Enable simplified DQN PRB allocator xApp",
)
parser.add_argument("--dqn-train", action="store_true", help="Train online (epsilon-greedy)")
parser.add_argument("--dqn-model", type=str, help="Path to load/save DQN model")
parser.add_argument("--dqn-epsilon-start", type=float, help="Epsilon start for exploration")
parser.add_argument("--dqn-epsilon-end", type=float, help="Epsilon end for exploration")
parser.add_argument("--dqn-epsilon-decay", type=int, help="Epsilon decay steps")
parser.add_argument("--dqn-period", type=int, help="Decision period in sim steps")
parser.add_argument("--dqn-move-step", type=int, help="PRBs moved per action")
parser.add_argument("--dqn-lr", type=float, help="Learning rate for the custom DQN")
parser.add_argument("--dqn-gamma", type=float, help="Discount factor for the custom DQN")
parser.add_argument("--dqn-target-update", type=int, help="Target network sync interval (steps)")
parser.add_argument(
    "--dqn-model-arch",
    choices=["mlp", "lstm", "tcn", "seq2seq"],
    help="Feature extractor for the custom DQN policy",
)
parser.add_argument("--dqn-seq-len", type=int, help="Sequence length for sequential DQN architectures")
parser.add_argument("--dqn-seq-hidden", type=int, help="Hidden size for sequential DQN architectures")
parser.add_argument(
    "--dqn-use-horizon",
    action="store_true",
    help="Aggregate states/rewards across the decision period before each DQN action",
)
parser.add_argument(
    "--dqn-horizon-reduce",
    choices=["mean", "last", "sum"],
    help="Reduction applied to horizon window for MLP agent inputs (default mean)",
)
parser.add_argument(
    "--dqn-save-interval",
    type=int,
    help="Checkpoint interval (steps) for custom DQN; 0 disables periodic saves",
)
parser.add_argument(
    "--dqn-device",
    type=str,
    help="Execution device for custom DQN (e.g., auto, cpu, cuda, cuda:0)",
)
parser.add_argument(
    "--dqn-log-interval",
    type=int,
    help="Interval (steps) for logging scalars to TensorBoard/W&B (default 1)",
)
parser.add_argument(
    "--dqn-aux-future-state",
    action="store_true",
    help="Enable auxiliary head to predict future state (dual-head training)",
)
parser.add_argument(
    "--dqn-aux-weight",
    type=float,
    help="Loss weight for auxiliary future-state prediction (default 0.1)",
)
parser.add_argument(
    "--dqn-aux-horizon",
    type=int,
    help="Number of lookahead steps for the auxiliary future-state head (default 1)",
)
parser.add_argument(
    "--dqn-max-train-steps",
    type=int,
    help="Maximum number of training decision steps before stopping (default 0 = unlimited)",
)
parser.add_argument("--dqn-log-tb", action="store_true", help="Enable TensorBoard logging for DQN")
parser.add_argument("--dqn-tb-dir", type=str, help="TensorBoard log dir for DQN")
parser.add_argument("--dqn-wandb", action="store_true", help="Enable Weights & Biases logging for DQN")
parser.add_argument("--dqn-wandb-project", type=str, help="W&B project name")
parser.add_argument("--dqn-wandb-name", type=str, help="W&B run name")
parser.add_argument(
    "--dqn-prb-episodic",
    action="store_true",
    help="Enable episodic variant of the DQN PRB allocator",
)
parser.add_argument(
    "--dqn-episode-config",
    type=str,
    help="Path to a JSON file describing episodic UE/trace configurations",
)
parser.add_argument(
    "--dqn-episode-config-json",
    type=str,
    help="Inline JSON blob describing episodic configurations (alternative to --dqn-episode-config)",
)
parser.add_argument(
    "--dqn-episode-loop",
    action="store_true",
    help="Loop episodic scenarios instead of stopping after the last one",
)
parser.add_argument(
    "--dqn-episode-len",
    type=int,
    help="Pseudo-episode length (steps) for DQN PRB allocator; 0 disables windowing",
)
parser.add_argument(
    "--dqn-episode-no-done",
    action="store_true",
    help="Do not mark pseudo-episode boundaries as terminal transitions",
)
# SB3 DQN variant controls
parser.add_argument("--sb3-dqn-prb", action="store_true", help="Enable SB3-based DQN PRB allocator xApp")
parser.add_argument("--sb3-dqn-model", type=str, help="Path to load/save SB3 DQN model")
parser.add_argument("--sb3-dqn-total-steps", type=int, help="Nominal total steps for SB3 progress schedule")
parser.add_argument("--sb3-dqn-target-update", type=int, help="Target network update interval for SB3 DQN")
parser.add_argument("--sb3-dqn-save-interval", type=int, help="Autosave interval (steps) for SB3 DQN model")
# Gym-style PRB allocator
parser.add_argument("--prb-gym", action="store_true", help="Enable Gym-style PRB allocator xApp")
parser.add_argument("--prb-gym-config", type=str, help="Episode config JSON for the Gym-style PRB xApp")
parser.add_argument("--prb-gym-loop", action="store_true", help="Loop the Gym-style episode catalog")
parser.add_argument(
    "--prb-gym-eps-decay",
    type=int,
    help="Per-episode epsilon decay steps for the Gym-style PRB allocator",
)
# Trace replay options
parser.add_argument(
    "--trace-speedup",
    type=float,
    default=None,
    help="Time scaling for trace playback (1.0 = realtime)",
)
parser.add_argument(
    "--trace-raw-map",
    action="append",
    help="Attach raw packet CSV to a UE: IMSI:file.csv:UE_IP (repeatable; UE_IP required)",
)
parser.add_argument(
    "--trace-bin",
    type=float,
    default=None,
    help="Bin size (seconds) for raw packet CSV aggregation (default 1.0)",
)
parser.add_argument(
    "--trace-overhead-bytes",
    type=int,
    default=None,
    help="Subtract this many bytes per packet in raw CSV (default 0)",
)
parser.add_argument(
    "--strict-real-traffic",
    action="store_true",
    help="Only show served traffic from traces (no fallback achievable rate)",
)
parser.add_argument(
    "--trace-validate-only",
    action="store_true",
    help="Validate configured trace files and exit",
)
parser.add_argument(
    "--trace-debug",
    action="store_true",
    help="Enable verbose trace replay debugging",
)
parser.add_argument(
    "--trace-debug-imsi",
    type=str,
    help="Comma-separated IMSIs to debug (if omitted, all)",
)
parser.add_argument(
    "--trace-loop",
    action="store_true",
    help="Loop trace playback (replay continuously)",
)
parser.add_argument(
    "--slice-prb",
    action="append",
    help="Initial downlink PRB quota per slice (format SLICE=PRBs or SLICE:PRBs). Repeatable.",
)
args, unknown = parser.parse_known_args()

if args.preset:
    os.environ["RAN_TOPOLOGY_PRESET"] = args.preset

if args.ws_host:
    os.environ["WS_SERVER_HOST"] = args.ws_host
if args.ws_port is not None:
    os.environ["WS_SERVER_PORT"] = str(args.ws_port)
if args.dash_port is not None:
    os.environ["DASH_PORT"] = str(args.dash_port)

# If per-slice counts are provided in simple preset, align total UEs accordingly
slice_counts = [c for c in (args.ue_embb, args.ue_urllc, args.ue_mmtc) if c is not None]
if slice_counts and (args.preset == "simple"):
    total_from_slices = sum(slice_counts)
    if args.ue_max is not None and args.ue_max != total_from_slices:
        print(
            f"[main] --ue-max ({args.ue_max}) != sum of slice counts ({total_from_slices}); adjusting total to {total_from_slices}."
        )
    os.environ["UE_DEFAULT_MAX_COUNT"] = str(total_from_slices)
elif args.ue_max is not None:
    os.environ["UE_DEFAULT_MAX_COUNT"] = str(args.ue_max)

if args.ue_embb is not None:
    os.environ["UE_SIMPLE_COUNT_EMBB"] = str(args.ue_embb)
if args.ue_urllc is not None:
    os.environ["UE_SIMPLE_COUNT_URLLC"] = str(args.ue_urllc)
if args.ue_mmtc is not None:
    os.environ["UE_SIMPLE_COUNT_MMTC"] = str(args.ue_mmtc)
if args.freeze_mobility:
    os.environ["SIM_FREEZE_MOBILITY"] = "1"
if args.sim_step is not None:
    os.environ["SIM_STEP_TIME_DEFAULT"] = str(args.sim_step)

# KPI dashboard env exports
if args.kpi_max_points is not None:
    os.environ["RAN_KPI_MAX_POINTS"] = str(args.kpi_max_points)
if args.kpi_history:
    os.environ["RAN_KPI_HISTORY_ENABLE"] = "1"
if args.kpi_log:
    os.environ["RAN_KPI_LOG_ENABLE"] = "1"
if args.kpi_log_dir:
    os.environ["RAN_KPI_LOG_DIR"] = args.kpi_log_dir

# DQN env exports
reload_settings = False
if args.dqn_train:
    os.environ["DQN_PRB_TRAIN"] = "1"
    reload_settings = True
else:
    os.environ["DQN_PRB_TRAIN"] = "0"
    reload_settings = True
if args.dqn_prb:
    os.environ["DQN_PRB_ENABLE"] = "1"
if args.dqn_prb_simple:
    os.environ["DQN_PRB_ENABLE"] = "1"
    os.environ["DQN_PRB_SIMPLE_ENABLE"] = "1"
if args.dqn_model:
    os.environ["DQN_PRB_MODEL_PATH"] = args.dqn_model
if args.dqn_epsilon_start is not None:
    os.environ["DQN_PRB_EPSILON_START"] = str(args.dqn_epsilon_start)
if args.dqn_epsilon_end is not None:
    os.environ["DQN_PRB_EPSILON_END"] = str(args.dqn_epsilon_end)
if args.dqn_epsilon_decay is not None:
    os.environ["DQN_PRB_EPSILON_DECAY"] = str(args.dqn_epsilon_decay)
if args.dqn_period is not None:
    os.environ["DQN_PRB_DECISION_PERIOD_STEPS"] = str(args.dqn_period)
if args.dqn_move_step is not None:
    os.environ["DQN_PRB_MOVE_STEP"] = str(args.dqn_move_step)
if args.dqn_model_arch is not None:
    os.environ["DQN_PRB_MODEL_ARCH"] = args.dqn_model_arch
if args.dqn_seq_len is not None:
    value = max(1, args.dqn_seq_len)
    os.environ["DQN_PRB_SEQ_LEN"] = str(value)
    reload_settings = True
if args.dqn_seq_hidden is not None:
    value = max(1, args.dqn_seq_hidden)
    os.environ["DQN_PRB_SEQ_HIDDEN"] = str(value)
    reload_settings = True
if args.dqn_use_horizon:
    os.environ["DQN_PRB_USE_DECISION_HORIZON"] = "1"
    reload_settings = True
if args.dqn_horizon_reduce:
    os.environ["DQN_PRB_HORIZON_REDUCE"] = args.dqn_horizon_reduce
    reload_settings = True
if args.dqn_save_interval is not None:
    os.environ["DQN_PRB_SAVE_INTERVAL"] = str(max(0, args.dqn_save_interval))
if args.dqn_device:
    os.environ["DQN_PRB_DEVICE"] = args.dqn_device
if args.dqn_log_interval is not None:
    os.environ["DQN_PRB_LOG_INTERVAL"] = str(max(1, args.dqn_log_interval))
if args.dqn_aux_future_state:
    os.environ["DQN_PRB_AUX_FUTURE_STATE"] = "1"
    reload_settings = True
if args.dqn_aux_weight is not None:
    value = max(0.0, args.dqn_aux_weight)
    os.environ["DQN_PRB_AUX_WEIGHT"] = str(value)
    reload_settings = True
if args.dqn_aux_horizon is not None:
    value = max(1, args.dqn_aux_horizon)
    os.environ["DQN_PRB_AUX_HORIZON"] = str(value)
    reload_settings = True
if args.dqn_max_train_steps is not None:
    value = max(0, args.dqn_max_train_steps)
    os.environ["DQN_PRB_MAX_TRAIN_STEPS"] = str(value)
    reload_settings = True
if args.dqn_lr is not None:
    os.environ["DQN_PRB_LR"] = str(max(1e-9, float(args.dqn_lr)))
if args.dqn_gamma is not None:
    os.environ["DQN_PRB_GAMMA"] = str(float(args.dqn_gamma))
if args.dqn_target_update is not None:
    os.environ["DQN_PRB_TARGET_UPDATE"] = str(max(1, int(args.dqn_target_update)))

_pending_settings_reload = reload_settings
if args.dqn_episode_len is not None:
    os.environ["DQN_PRB_EPISODE_LEN"] = str(max(0, args.dqn_episode_len))
if args.dqn_episode_no_done:
    os.environ["DQN_PRB_EPISODE_MARK_DONE"] = "0"
if args.dqn_log_tb:
    os.environ["DQN_TB_ENABLE"] = "1"
if args.dqn_tb_dir:
    os.environ["DQN_TB_DIR"] = args.dqn_tb_dir
if args.dqn_wandb:
    os.environ["DQN_WANDB_ENABLE"] = "1"
if args.dqn_wandb_project:
    os.environ["DQN_WANDB_PROJECT"] = args.dqn_wandb_project
if args.dqn_wandb_name:
    os.environ["DQN_WANDB_RUNNAME"] = args.dqn_wandb_name
if args.dqn_prb_episodic:
    os.environ["DQN_PRB_EPISODIC_ENABLE"] = "1"
if args.dqn_episode_config:
    os.environ["DQN_EPISODE_CONFIG_PATH"] = args.dqn_episode_config
if args.dqn_episode_config_json:
    os.environ["DQN_EPISODE_CONFIG_JSON"] = args.dqn_episode_config_json
if args.dqn_episode_loop:
    os.environ["DQN_EPISODE_LOOP"] = "1"

# SB3 DQN env exports
if args.sb3_dqn_prb:
    os.environ["SB3_DQN_PRB_ENABLE"] = "1"
if args.sb3_dqn_model:
    os.environ["SB3_DQN_MODEL_PATH"] = args.sb3_dqn_model
if args.sb3_dqn_total_steps is not None:
    os.environ["SB3_DQN_TOTAL_STEPS"] = str(args.sb3_dqn_total_steps)
if args.sb3_dqn_target_update is not None:
    os.environ["SB3_DQN_TARGET_UPDATE"] = str(args.sb3_dqn_target_update)
if args.sb3_dqn_save_interval is not None:
    os.environ["SB3_DQN_SAVE_INTERVAL"] = str(args.sb3_dqn_save_interval)
# Gym-style xApp env exports
if args.prb_gym:
    os.environ["PRB_GYM_ENABLE"] = "1"
if args.prb_gym_config:
    os.environ["PRB_GYM_CONFIG_PATH"] = args.prb_gym_config
if args.prb_gym_loop:
    os.environ["PRB_GYM_LOOP"] = "1"
if args.prb_gym_eps_decay is not None:
    try:
        _pg_decay = max(1, int(args.prb_gym_eps_decay))
    except Exception:
        _pg_decay = None
    if _pg_decay:
        os.environ["PRB_GYM_EPS_DECAY_PER_EPISODE"] = str(_pg_decay)

# Trace mapping and options (export via env before importing settings)
if args.trace_speedup is not None:
    os.environ["TRACE_SPEEDUP"] = str(args.trace_speedup)
if args.strict_real_traffic:
    os.environ["STRICT_REAL_TRAFFIC"] = "1"
if args.trace_raw_map:
    raw_items = []
    for item in args.trace_raw_map:
        # Supported formats:
        #   IMSI:file.csv:UE_IP
        #   ALL:file.csv:UE_IP  (apply to all UEs)
        #   slice:<NAME>:file.csv:UE_IP  (apply to UEs of a slice)
        if not item or ":" not in item:
            continue
        parts = item.split(":")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        # Build mapping dict based on first token
        entry = None
        if key.upper() in ("ALL", "*"):
            # ALL:file:UE_IP
            if len(parts) < 3:
                continue
            ue_ip = parts[-1].strip()
            path = ":".join(parts[1:-1]).strip()
            if not (path and ue_ip):
                continue
            entry = {"imsi": "*", "file": path, "ue_ip": ue_ip}
        elif key.lower() == "slice" or key.lower().startswith("slice:"):
            # Two supported forms:
            #  - slice:NAME:file:UE_IP (split as ['slice', 'NAME', 'file', 'UE_IP'])
            #  - slice:NAME:file:UE_IP (split as ['slice:NAME', 'file', 'UE_IP'])
            if key.lower() == "slice":
                if len(parts) < 4:
                    continue
                slice_name = parts[1].strip()
                ue_ip = parts[-1].strip()
                path = ":".join(parts[2:-1]).strip()
            else:
                # key like 'slice:NAME'
                slice_name = key.split(":", 1)[1].strip() if ":" in key else ""
                if len(parts) < 3:
                    continue
                ue_ip = parts[-1].strip()
                path = ":".join(parts[1:-1]).strip()
            if not (slice_name and path and ue_ip):
                continue
            entry = {"slice": slice_name, "file": path, "ue_ip": ue_ip}
        else:
            # IMSI:file:UE_IP
            if len(parts) < 3:
                continue
            ue_ip = parts[-1].strip()
            path = ":".join(parts[1:-1]).strip()
            if not (path and ue_ip):
                continue
            entry = {"imsi": key, "file": path, "ue_ip": ue_ip}
        if entry is None:
            continue
        raw_items.append(entry)
    if raw_items:
        os.environ["TRACE_RAW_MAP_JSON"] = json.dumps(raw_items)
if args.trace_bin is not None:
    os.environ["TRACE_BIN"] = str(args.trace_bin)
    os.environ["TRACE_BIN_OVERRIDE"] = "1"
if args.trace_overhead_bytes is not None:
    os.environ["TRACE_OVERHEAD_BYTES"] = str(args.trace_overhead_bytes)
if args.trace_debug:
    os.environ["TRACE_DEBUG"] = "1"
if args.trace_debug_imsi:
    os.environ["TRACE_DEBUG_IMSI"] = args.trace_debug_imsi
if args.trace_loop:
    os.environ["TRACE_LOOP"] = "1"

if args.slice_prb:
    overrides = {}
    for item in args.slice_prb:
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        key = key.strip()
        if not key:
            continue
        try:
            overrides[key] = int(float(value))
        except Exception:
            continue
    if overrides:
        os.environ["RAN_SLICE_DL_PRB_OVERRIDES"] = json.dumps(overrides)

# Load .env (won't override already-set env)
dotenv.load_dotenv()

from flask import logging
import websockets
import settings
from utils import (
    WebSocketResponse,
    handle_start_simulation,
    handle_stop_simulation,
    handle_get_simulation_state,
    handle_get_routes,
    handle_query_knowledge,
    stream_agent_chat,
    handle_network_user_action,
    setup_logging,
    WebSocketSingleton,
)
from network_layer.simulation_engine import SimulationEngine
from knowledge_layer import KnowledgeRouter
from intelligence_layer import engineer_chat_agent
from intelligence_layer.ai_service_pipeline import handle_ai_service_pipeline_chat

if _pending_settings_reload:
    settings = importlib.reload(settings)

#setup_logging(level=logging.WARNING)

# Validate configured traces (if any) early and log summary
try:
    from utils.traffic_trace import validate_traces_configuration
    import logging as _logging
    _tv_logger = _logging.getLogger("trace_validation")
    validate_traces_configuration(
        raw_map=getattr(settings, "TRACE_RAW_MAP", []),
        bin_s=getattr(settings, "TRACE_BIN", 1.0),
        overhead_bytes=getattr(settings, "TRACE_OVERHEAD_BYTES", 0),
        logger_name="trace_validation",
    )
except Exception:
    # Non-fatal; continue without validation if anything unexpected occurs here
    print("[main] Warning: Exception during trace validation; continuing without validation.")
    #pass


COMMAND_HANDLERS = {
    ("network_layer", "start_simulation"): handle_start_simulation,
    ("network_layer", "stop_simulation"): handle_stop_simulation,
    ("network_layer", "get_simulation_state"): handle_get_simulation_state,
    ("knowledge_layer", "get_routes"): handle_get_routes,
    ("knowledge_layer", "query_knowledge"): handle_query_knowledge,
    ("intelligence_layer", "ai_service_pipeline"): handle_ai_service_pipeline_chat,
    ("intelligence_layer", "network_engineer_chat"): partial(
        stream_agent_chat,
        command="network_engineer_chat_response",
        agent_func=engineer_chat_agent,
    ),
    ("intelligence_layer", "network_user_action"): handle_network_user_action,
}


async def websocket_handler(websocket):
    WebSocketSingleton().set_websocket(websocket)
    simulation_engine = SimulationEngine()
    simulation_engine.reset_network()
    simulation_engine.network_setup()
    knowledge_router = KnowledgeRouter()
    knowledge_router.import_routes(simulation_engine)
    while True:
        message = await websocket.recv()
        try:
            message_json = json.loads(message)
            layer = message_json.get("layer")
            command = message_json.get("command")
            data = message_json.get("data", {})
        except (json.JSONDecodeError, KeyError):
            response = WebSocketResponse(
                layer=None, command=None, response=None, error="Invalid message format"
            )
            await websocket.send(response.to_json())
            continue

        try:
            handler = COMMAND_HANDLERS.get((layer, command))
            if handler:
                await handler(
                    websocket=websocket,
                    simulation_engine=simulation_engine,
                    knowledge_router=knowledge_router,
                    data=data,
                )
            else:
                response = WebSocketResponse(
                    layer=layer,
                    command=command,
                    response=None,
                    error=f"Unknown command: {command}",
                )
                await websocket.send(response.to_json())
        except Exception as e:
            response = WebSocketResponse(
                layer=layer, command=command, response=None, error=str(e)
            )
            await websocket.send(response.to_json())


async def main():
    # Optional: validate-only mode to isolate and view checks with no other output
    if args.trace_validate_only:
        return
    # Headless mode: run simulation loop without WebSocket server
    if args.mode == "headless":
        eng = SimulationEngine()
        eng.reset_network()
        eng.network_setup()
        for _ in range(args.steps):
            eng.sim_step += 1
            eng.step(settings.SIM_STEP_TIME_DEFAULT)
            await asyncio.sleep(settings.SIM_STEP_TIME_DEFAULT)
        print(f"Headless run completed. BS: {list(eng.base_station_list.keys())}, Cells: {list(eng.cell_list.keys())}")
        # Optional trace replay summary
        try:
            if getattr(settings, "TRACE_DEBUG", False):
                print("\n=== Trace Replay Summary ===")
                for ue in eng.ue_list.values():
                    if getattr(ue, "_trace_samples", None) is None:
                        continue
                    total_samples = len(ue._trace_samples)
                    total_dl = getattr(ue, "_trace_enqueued_dl_total", 0)
                    total_served = getattr(ue, "_trace_served_dl_total", 0)
                    buf = ue.dl_buffer_bytes
                    ok = (total_dl == total_served + buf)
                    status = "OK" if ok else "MISMATCH"
                    print(f"{ue.ue_imsi}: samples={total_samples} enq_dl={total_dl}B served_dl={total_served}B remaining_buf={buf}B [{status}]")
                print("===========================\n")
        except Exception:
            pass
        return

    # Server mode (default)
    async with websockets.serve(
        websocket_handler, settings.WS_SERVER_HOST, settings.WS_SERVER_PORT
    ):
        print(
            f"WebSocket server started on ws://{settings.WS_SERVER_HOST}:{settings.WS_SERVER_PORT}"
        )
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
