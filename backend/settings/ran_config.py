# ---------------------------
# RAN Configuration
# ---------------------------
import os
import json
RAN_BS_LOAD_HISTORY_LENGTH = 3
RAN_BS_REF_SIGNAL_DEFAULT_TRASNMIT_POWER = 40

# Usually n1 bands are often frequency-division duplex (FDD) bands, while n78 and n258 are typically time-division duplex (TDD) bands.
# so while n78 and n258 bands needs to be split into downlink and uplink separately, n1 bands are usually not split
# as they have a fixed downlink/uplink ratio linked to separate carrier frequencies
# but for simplicity, we use the same split mechanism for all bands.
RAN_CELL_DL_UL_PRB_SPLIT = {
    "n1": (0.7, 0.3),
    "n78": (0.8, 0.2),
    "n258": (0.9, 0.1),
}

RAN_SYMBOL_DURATION = 71.4e-6  # 71.4 µs per OFDM symbol
RAN_SYMBOLS_PER_SLOT = 14
RAN_SLOT_DURATION = RAN_SYMBOL_DURATION * RAN_SYMBOLS_PER_SLOT  # ~1 ms
RAN_SUBCARRIERS_PER_PRB = 12
RAN_SUBCARRIER_SPACING_Hz = 15e3  # 15 kHz
RAN_PRB_BANDWIDTH_Hz = RAN_SUBCARRIERS_PER_PRB * RAN_SUBCARRIER_SPACING_Hz  # 180 kHz
RAN_RESOURCE_ELEMENTS_PER_PRB_PER_SLOT = (
    RAN_SUBCARRIERS_PER_PRB * RAN_SYMBOLS_PER_SLOT
)  # 12 subcarriers * 14 symbols = 168 resource elements per PRB per slot

# Live KPI Dashboard controls
# 0 means unlimited per-UE DL PRB cap; UI will map 0 -> unlimited.
RAN_PRB_CAP_SLIDER_DEFAULT = 0
RAN_PRB_CAP_SLIDER_MAX = 50

# Static per-slice DL PRB split (fractions sum to <= 1.0). Remaining PRBs, if any, stay unused.
# These are applied per cell (converted to integer PRB counts using that cell's max_dl_prb).
RAN_SLICE_DL_PRB_SPLIT_DEFAULT = {
    "eMBB": 0.7,
    "URLLC": 0.2,
    "mMTC": 0.1,
}

_prb_override_env = os.getenv("RAN_SLICE_DL_PRB_OVERRIDES", "")
try:
    if _prb_override_env:
        _prb_override_dict = json.loads(_prb_override_env)
        RAN_SLICE_DL_PRB_OVERRIDES = {
            str(k): int(float(v)) for k, v in _prb_override_dict.items()
        }
    else:
        RAN_SLICE_DL_PRB_OVERRIDES = {}
except Exception:
    RAN_SLICE_DL_PRB_OVERRIDES = {}

# UI knob step for slice split sliders (set small for near‑continuous control)
RAN_SLICE_KNOB_STEP_FRAC = 0.001

# KPI dashboard history and logging
# Rolling window length for live plots (points kept in memory). Set 0 for unbounded.
try:
    RAN_KPI_MAX_POINTS = int(os.getenv("RAN_KPI_MAX_POINTS", "50"))
except Exception:
    RAN_KPI_MAX_POINTS = 50

# Enable CSV logging of KPIs each simulation step
RAN_KPI_LOG_ENABLE = os.getenv("RAN_KPI_LOG_ENABLE", "0") in ("1", "true", "True")
# Output directory for KPI logs
RAN_KPI_LOG_DIR = os.getenv("RAN_KPI_LOG_DIR", "backend/kpi_logs")

# Enable interactive history navigation (range slider) on KPI charts
RAN_KPI_HISTORY_ENABLE = os.getenv("RAN_KPI_HISTORY_ENABLE", "0") in ("1", "true", "True")

# ETSI TS 138 214 V15.3.0. Release 15 Table 5.1.3.1-2 MCS index table 2 for PDSCH
# MCS Index (I_MCS) | Modulation Order Qm | Target code Rate R x [1024] | Spectral efficiency
RAN_MCS_SPECTRAL_EFFICIENCY_TABLE = {
    0: {"modulation_order": 2, "target_code_rate": 120, "spectral_efficiency": 0.2344},
    1: {"modulation_order": 2, "target_code_rate": 193, "spectral_efficiency": 0.3770},
    2: {"modulation_order": 2, "target_code_rate": 308, "spectral_efficiency": 0.6016},
    3: {"modulation_order": 2, "target_code_rate": 449, "spectral_efficiency": 0.8770},
    4: {"modulation_order": 2, "target_code_rate": 602, "spectral_efficiency": 1.1758},
    5: {"modulation_order": 4, "target_code_rate": 378, "spectral_efficiency": 1.4766},
    6: {"modulation_order": 4, "target_code_rate": 434, "spectral_efficiency": 1.6953},
    7: {"modulation_order": 4, "target_code_rate": 490, "spectral_efficiency": 1.9141},
    8: {"modulation_order": 4, "target_code_rate": 553, "spectral_efficiency": 2.1602},
    9: {"modulation_order": 4, "target_code_rate": 616, "spectral_efficiency": 2.4063},
    10: {"modulation_order": 4, "target_code_rate": 658, "spectral_efficiency": 2.5703},
    11: {"modulation_order": 6, "target_code_rate": 466, "spectral_efficiency": 2.7305},
    12: {"modulation_order": 6, "target_code_rate": 517, "spectral_efficiency": 3.0293},
    13: {"modulation_order": 6, "target_code_rate": 567, "spectral_efficiency": 3.3223},
    14: {"modulation_order": 6, "target_code_rate": 616, "spectral_efficiency": 3.6094},
    15: {"modulation_order": 6, "target_code_rate": 666, "spectral_efficiency": 3.9023},
    16: {"modulation_order": 6, "target_code_rate": 719, "spectral_efficiency": 4.2129},
    17: {"modulation_order": 6, "target_code_rate": 772, "spectral_efficiency": 4.5234},
    18: {"modulation_order": 6, "target_code_rate": 822, "spectral_efficiency": 4.8164},
    19: {"modulation_order": 6, "target_code_rate": 873, "spectral_efficiency": 5.1152},
    20: {
        "modulation_order": 8,
        "target_code_rate": 682.5,
        "spectral_efficiency": 5.3320,
    },
    21: {"modulation_order": 8, "target_code_rate": 711, "spectral_efficiency": 5.5547},
    22: {"modulation_order": 8, "target_code_rate": 754, "spectral_efficiency": 5.8906},
    23: {"modulation_order": 8, "target_code_rate": 797, "spectral_efficiency": 6.2266},
    24: {"modulation_order": 8, "target_code_rate": 841, "spectral_efficiency": 6.5703},
    25: {"modulation_order": 8, "target_code_rate": 885, "spectral_efficiency": 6.9141},
    26: {
        "modulation_order": 8,
        "target_code_rate": 916.5,
        "spectral_efficiency": 7.1602,
    },
    27: {"modulation_order": 8, "target_code_rate": 948, "spectral_efficiency": 7.4063},
    # 28: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 29: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 30: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 31: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
}


# ---------------------------
# DQN/Gym allocator settings
# ---------------------------
# Steps between actions (period in sim steps)
try:
    DQN_PRB_DECISION_PERIOD_STEPS = int(os.getenv("DQN_PRB_DECISION_PERIOD_STEPS", "1"))
except Exception:
    DQN_PRB_DECISION_PERIOD_STEPS = 1
# PRBs moved per action (Table 3 uses 1 RB; treat RB=PRB here)
try:
    DQN_PRB_MOVE_STEP = int(os.getenv("DQN_PRB_MOVE_STEP", "1"))
except Exception:
    DQN_PRB_MOVE_STEP = 1

# DQN hyperparameters
try:
    DQN_PRB_EPSILON_START = float(os.getenv("DQN_PRB_EPSILON_START", "1.0"))
    DQN_PRB_EPSILON_END = float(os.getenv("DQN_PRB_EPSILON_END", "0.1"))
    DQN_PRB_EPSILON_DECAY = int(os.getenv("DQN_PRB_EPSILON_DECAY", "50000"))
    DQN_PRB_GAMMA = float(os.getenv("DQN_PRB_GAMMA", "0.99"))
    DQN_PRB_LR = float(os.getenv("DQN_PRB_LR", "1e-2"))
    DQN_PRB_BATCH = int(os.getenv("DQN_PRB_BATCH", "64"))
    DQN_PRB_BUFFER = int(os.getenv("DQN_PRB_BUFFER", "100000"))
except Exception:
    DQN_PRB_EPSILON_START, DQN_PRB_EPSILON_END, DQN_PRB_EPSILON_DECAY = 1.0, 0.1, 50000
    DQN_PRB_GAMMA, DQN_PRB_LR, DQN_PRB_BATCH, DQN_PRB_BUFFER = 0.99, 1e-2, 64, 100000
try:
    DQN_PRB_TARGET_UPDATE = int(os.getenv("DQN_PRB_TARGET_UPDATE", "200"))
except Exception:
    DQN_PRB_TARGET_UPDATE = 200

DQN_PRB_MODEL_ARCH = os.getenv("DQN_PRB_MODEL_ARCH", "mlp").strip().lower()
try:
    DQN_PRB_SEQ_LEN = int(os.getenv("DQN_PRB_SEQ_LEN", "1"))
except Exception:
    DQN_PRB_SEQ_LEN = 1
try:
    DQN_PRB_SEQ_HIDDEN = int(os.getenv("DQN_PRB_SEQ_HIDDEN", "128"))
except Exception:
    DQN_PRB_SEQ_HIDDEN = 128

try:
    DQN_PRB_SAVE_INTERVAL = int(os.getenv("DQN_PRB_SAVE_INTERVAL", "0"))
except Exception:
    DQN_PRB_SAVE_INTERVAL = 0

DQN_PRB_DEVICE = os.getenv("DQN_PRB_DEVICE", "auto")

try:
    DQN_PRB_LOG_INTERVAL = int(os.getenv("DQN_PRB_LOG_INTERVAL", "1"))
except Exception:
    DQN_PRB_LOG_INTERVAL = 1

try:
    DQN_PRB_MAX_TRAIN_STEPS = int(os.getenv("DQN_PRB_MAX_TRAIN_STEPS", "0"))
except Exception:
    DQN_PRB_MAX_TRAIN_STEPS = 0

# Reward shaping weights and parameters
try:
    DQN_WEIGHT_EMBB = float(os.getenv("DQN_WEIGHT_EMBB", "0.5"))
    DQN_WEIGHT_URLLC = float(os.getenv("DQN_WEIGHT_URLLC", "0.5"))
    DQN_URLLC_GAMMA_S = float(os.getenv("DQN_URLLC_GAMMA_S", "0.01"))  # 10 ms
    DQN_NEED_SATURATION = float(os.getenv("DQN_NEED_SATURATION", "1.5"))
except Exception:
    DQN_WEIGHT_EMBB, DQN_WEIGHT_URLLC = 0.5, 0.5
    DQN_URLLC_GAMMA_S = 0.01
    DQN_NEED_SATURATION = 1.5

# Model path
DQN_PRB_MODEL_PATH = os.getenv("DQN_PRB_MODEL_PATH", "backend/models/dqn_prb.pt")

# DQN logging/telemetry
DQN_TB_ENABLE = os.getenv("DQN_TB_ENABLE", "0") in ("1", "true", "True")
DQN_TB_DIR = os.getenv("DQN_TB_DIR", "backend/tb_logs")
PRB_GYM_ENABLE = os.getenv("PRB_GYM_ENABLE", "0") in ("1", "true", "True")
PRB_GYM_CONFIG_PATH = os.getenv("PRB_GYM_CONFIG_PATH", "")
PRB_GYM_LOOP = os.getenv("PRB_GYM_LOOP", "0") in ("1", "true", "True")
PRB_GYM_SHUFFLE_EPISODES = os.getenv("PRB_GYM_SHUFFLE_EPISODES", "0") in ("1", "true", "True")
PRB_GYM_TRAIN = os.getenv("PRB_GYM_TRAIN", "1") in ("1", "true", "True")
PRB_GYM_EVAL_ONLY = os.getenv("PRB_GYM_EVAL_ONLY", "0") in ("1", "true", "True")
PRB_GYM_LOAD_MODEL_PATH = os.getenv("PRB_GYM_LOAD_MODEL_PATH", "")
try:
    _gym_decay_raw = os.getenv("PRB_GYM_EPS_DECAY_STEPS")
    if _gym_decay_raw is None:
        _gym_decay_raw = os.getenv("PRB_GYM_EPS_DECAY_PER_EPISODE", "2000")
    PRB_GYM_EPS_DECAY_STEPS = int(_gym_decay_raw)
except Exception:
    PRB_GYM_EPS_DECAY_STEPS = 2000
# Backward compatibility for existing configs/tests
PRB_GYM_EPS_DECAY_PER_EPISODE = PRB_GYM_EPS_DECAY_STEPS
PRB_CONST_ALLOC_ENABLE = os.getenv("PRB_CONST_ALLOC_ENABLE", "0") in ("1", "true", "True")
def _const_alloc_env(key):
    val = os.getenv(key)
    if val is None:
        return None
    try:
        return max(0, int(val))
    except Exception:
        return None
PRB_CONST_ALLOC_MAP = {}
_embb_name = locals().get("NETWORK_SLICE_EMBB_NAME", "eMBB")
_urllc_name = locals().get("NETWORK_SLICE_URLLC_NAME", "URLLC")
_embb_val = _const_alloc_env("PRB_CONST_ALLOC_EMBB")
_urllc_val = _const_alloc_env("PRB_CONST_ALLOC_URLLC")
if _embb_val is not None:
    PRB_CONST_ALLOC_MAP[_embb_name] = _embb_val
if _urllc_val is not None:
    PRB_CONST_ALLOC_MAP[_urllc_name] = _urllc_val
try:
    PRB_CONST_LOG_INTERVAL = max(1, int(os.getenv("PRB_CONST_LOG_INTERVAL", "500")))
except Exception:
    PRB_CONST_LOG_INTERVAL = 500


RAN_TOPOLOGY_PRESET = os.getenv("RAN_TOPOLOGY_PRESET", "default")  # 'default' or 'simple'


def RAN_BS_DEFAULT_CELLS(bs_id):
    """Return the default cell list for a base station.

    If RAN_TOPOLOGY_PRESET == 'simple', return a single n78 cell for easier testing.
    Otherwise, return the mid and high frequency cells.
    """
    if RAN_TOPOLOGY_PRESET == "simple":
        return [
            {
                "cell_id": f"{bs_id}_cell_mid_freq",
                "frequency_band": "n78",
                "carrier_frequency_MHz": 3500,
                "bandwidth_Hz": 100e6,
                "max_prb": 273,
                "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
                "max_ul_prb": 273 - int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
                "cell_radius": 800,
                "transmit_power_dBm": 40,
                "cell_individual_offset_dBm": 0,
                "frequency_priority": 5,
                "qrx_level_min": -98,
            },
        ]
    # default: two cells
    return [
        {
            "cell_id": f"{bs_id}_cell_mid_freq",
            "frequency_band": "n78",
            "carrier_frequency_MHz": 3500,
            "bandwidth_Hz": 100e6,
            "max_prb": 273,
            "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
            "max_ul_prb": 273 - int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
            "cell_radius": 800,
            "transmit_power_dBm": 40,
            "cell_individual_offset_dBm": 0,
            "frequency_priority": 5,
            "qrx_level_min": -98,
        },
        {
            "cell_id": f"{bs_id}_cell_high_freq",
            "frequency_band": "n258",
            "carrier_frequency_MHz": 26000,
            "bandwidth_Hz": 400e6,
            "max_prb": 264,
            "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n258"][0] * 264),
            "max_ul_prb": 264 - int(RAN_CELL_DL_UL_PRB_SPLIT["n258"][0] * 264),
            "cell_radius": 300,
            "transmit_power_dBm": 50,  # assume achieved by beamforming
            "cell_individual_offset_dBm": 13,
            "frequency_priority": 7,
            "qrx_level_min": -89,
        },
    ]


# Event	Description	Condition
# A1	Serving becomes better than threshold	Serving > threshold
# A2	Serving becomes worse than threshold	Serving < threshold
# A3	Neighbor becomes offset better than serving	Neighbor > Serving + offset
# A4	Neighbor becomes better than threshold	Neighbor > threshold
# A5	Serving becomes worse than threshold1 AND neighbor becomes better than threshold2	Serving < threshold1 AND Neighbor > threshold2
# B1	Inter-RAT neighbor becomes better than threshold	Used for LTE/other RATs
# B2	Serving becomes worse than threshold1 AND Inter-RAT neighbor becomes better than threshold2	Also for Inter-RAT
def RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS():
    return [
        {
            "event_id": "A3",
            "power_threshold": 3,
            "time_to_trigger_in_sim_steps": 3,  # normally time-t-trigger or TTT is in time, but since we are in simulation, we can use simulation steps
        }
    ]


def RAN_BS_EDGE_DEFAULT_SERVER():
    return {
        "node_id": "LAP004262",
        "device_type": "DeviceType.CPU",
        "cpu_memory_GB": 10,  # set this to the docker CPU memory limit minus some overhead.
        "device_memory_GB": 0.0,  # this is the device (GPU, FPGA, etc.) memory, set to 0 if not applicable
    }


if RAN_TOPOLOGY_PRESET == "simple":
    RAN_DEFAULT_BS_LIST = [
        {
            "bs_id": "bs_1",
            "position_x": 1000,
            "position_y": 1000,
            "cell_list": RAN_BS_DEFAULT_CELLS("bs_1"),
            "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
            "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
        }
    ]
else:
    RAN_DEFAULT_BS_LIST = [
        {
            "bs_id": "bs_11",
            "position_x": 500,
            "position_y": 500,
            "cell_list": RAN_BS_DEFAULT_CELLS("bs_11"),
            "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
            "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
        },
        {
            "bs_id": "bs_12",
            "position_x": 1500,
            "position_y": 500,
            "cell_list": RAN_BS_DEFAULT_CELLS("bs_12"),
            "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
            "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
        },
        {
            "bs_id": "bs_21",
            "position_x": 500,
            "position_y": 1500,
            "cell_list": RAN_BS_DEFAULT_CELLS("bs_21"),
            "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
            "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
        },
        {
            "bs_id": "bs_22",
            "position_x": 1500,
            "position_y": 1500,
            "cell_list": RAN_BS_DEFAULT_CELLS("bs_22"),
            "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
            "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
        },
    ]
