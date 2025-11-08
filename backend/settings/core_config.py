import os
from . import slice_config as _slice_cfg
from .ue_config import UE_DEFAULT_MAX_COUNT

import random

NETWORK_SLICE_EMBB_NAME = _slice_cfg.NETWORK_SLICE_EMBB_NAME
NETWORK_SLICE_URLLC_NAME = _slice_cfg.NETWORK_SLICE_URLLC_NAME
NETWORK_SLICE_MTC_NAME = getattr(_slice_cfg, "NETWORK_SLICE_MTC_NAME", None)

CORE_UE_SUBSCRIPTION_DATA = {}

# Optional explicit counts for the simple preset
_preset = os.getenv("RAN_TOPOLOGY_PRESET", "default")
e_cnt = os.getenv("UE_SIMPLE_COUNT_EMBB")
u_cnt = os.getenv("UE_SIMPLE_COUNT_URLLC")
m_cnt = os.getenv("UE_SIMPLE_COUNT_MMTC")
try:
    e_cnt = int(e_cnt) if e_cnt is not None else None
    u_cnt = int(u_cnt) if u_cnt is not None else None
    m_cnt = int(m_cnt) if m_cnt is not None else None
except ValueError:
    e_cnt = u_cnt = m_cnt = None

if _preset == "simple" and any(v is not None for v in (e_cnt, u_cnt, m_cnt)):
    # Deterministic distribution: first m_cnt IMSIs are mMTC, next u_cnt URLLC, next e_cnt eMBB
    total = UE_DEFAULT_MAX_COUNT
    m = max(0, min(m_cnt or 0, total)) if NETWORK_SLICE_MTC_NAME else 0
    u = max(0, min(u_cnt or 0, max(0, total - m)))
    e = max(0, min(e_cnt or 0, max(0, total - m - u)))
    # Fill remainder with eMBB
    rem = max(0, total - (m + u + e))
    e += rem

    idx = 0
    if NETWORK_SLICE_MTC_NAME:
        for _ in range(m):
            CORE_UE_SUBSCRIPTION_DATA[f"IMSI_{idx}"] = [NETWORK_SLICE_MTC_NAME]
            idx += 1
    for _ in range(u):
        CORE_UE_SUBSCRIPTION_DATA[f"IMSI_{idx}"] = [NETWORK_SLICE_URLLC_NAME]
        idx += 1
    for _ in range(e):
        CORE_UE_SUBSCRIPTION_DATA[f"IMSI_{idx}"] = [NETWORK_SLICE_EMBB_NAME]
        idx += 1
else:
    # Default randomized distribution
    for i in range(UE_DEFAULT_MAX_COUNT):
        UE_IMSI = f"IMSI_{i}"

        # roughly 20% IoT (mMTC only)
        if NETWORK_SLICE_MTC_NAME and random.random() < 0.2:
            CORE_UE_SUBSCRIPTION_DATA[UE_IMSI] = [NETWORK_SLICE_MTC_NAME]
            continue

        # general UEs default to eMBB
        CORE_UE_SUBSCRIPTION_DATA[UE_IMSI] = [NETWORK_SLICE_EMBB_NAME]

        # 50% chance they also subscribe to URLLC
        if random.random() < 0.5:
            CORE_UE_SUBSCRIPTION_DATA[UE_IMSI].append(NETWORK_SLICE_URLLC_NAME)
