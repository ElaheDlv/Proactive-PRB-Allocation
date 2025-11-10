from .network_infra_config import NETWORK_COVERAGE_HEIGHT, NETWORK_COVERAGE_WIDTH
import os

# ---------------------------
# User Equipment (UE) Configuration
# ---------------------------
UE_BOUNDARY_X_MIN = 0
UE_BOUNDARY_X_MAX = NETWORK_COVERAGE_WIDTH
UE_BOUNDARY_Y_MIN = 0
UE_BOUNDARY_Y_MAX = NETWORK_COVERAGE_HEIGHT
UE_DEFAULT_TIMEOUT = 20000

# Simple topology option via env: RAN_TOPOLOGY_PRESET=simple
_PRESET = os.getenv("RAN_TOPOLOGY_PRESET", "default")

if _PRESET == "simple":
    UE_DEFAULT_SPAWN_RATE_MIN = 1
    UE_DEFAULT_SPAWN_RATE_MAX = 2
    UE_DEFAULT_MAX_COUNT = int(os.getenv("UE_DEFAULT_MAX_COUNT", 10))
else:
    UE_DEFAULT_SPAWN_RATE_MIN = 1
    UE_DEFAULT_SPAWN_RATE_MAX = 5
    UE_DEFAULT_MAX_COUNT = int(os.getenv("UE_DEFAULT_MAX_COUNT", 50))

UE_speed_mps_MIN = 10
UE_speed_mps_MAX = 20
UE_SERVING_CELL_HISTORY_LENGTH = 10
UE_SSB_DETECTION_THRESHOLD = -110
UE_TRANSMIT_POWER = 23
UE_TEMPERATURE_K = 290
UE_AI_SERVICE_REQUEST_COUNTDOWN = 10

# 3GPP TS 38.214 version 15.3.0 Release 15
# Table 5.2.2.1-3: 4-bit CQI Table 2
UE_CQI_MCS_SPECTRAL_EFFICIENCY_TABLE = {
    0: {
        "modulation": None,  # out of range
        "code_rate": 0,
        "spectral_efficiency": 0,
    },
    1: {"modulation": "QPSK", "code_rate": 78, "spectral_efficiency": 0.1523},
    2: {"modulation": "QPSK", "code_rate": 193, "spectral_efficiency": 0.3770},
    3: {"modulation": "QPSK", "code_rate": 449, "spectral_efficiency": 0.8770},
    4: {"modulation": "16QAM", "code_rate": 378, "spectral_efficiency": 1.4766},
    5: {"modulation": "16QAM", "code_rate": 490, "spectral_efficiency": 1.9141},
    6: {"modulation": "16QAM", "code_rate": 616, "spectral_efficiency": 2.4063},
    7: {"modulation": "64QAM", "code_rate": 466, "spectral_efficiency": 2.7305},
    8: {"modulation": "64QAM", "code_rate": 567, "spectral_efficiency": 3.3223},
    9: {"modulation": "64QAM", "code_rate": 666, "spectral_efficiency": 3.9023},
    10: {"modulation": "64QAM", "code_rate": 772, "spectral_efficiency": 4.5234},
    11: {"modulation": "64QAM", "code_rate": 873, "spectral_efficiency": 5.1152},
    12: {"modulation": "256QAM", "code_rate": 711, "spectral_efficiency": 5.5547},
    13: {"modulation": "256QAM", "code_rate": 797, "spectral_efficiency": 6.2266},
    14: {"modulation": "256QAM", "code_rate": 885, "spectral_efficiency": 6.9141},
    15: {"modulation": "256QAM", "code_rate": 948, "spectral_efficiency": 7.4063},
}
