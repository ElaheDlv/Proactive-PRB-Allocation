# ---------------------------
# Network Slice Configuration
# ---------------------------
NETWORK_SLICE_EMBB_NAME = "eMBB"
NETWORK_SLICE_URLLC_NAME = "URLLC"
NETWORK_SLICE_MTC_NAME = "mMTC"

NETWORK_SLICES = {
    NETWORK_SLICE_EMBB_NAME: {
        "5QI": 9,
        "GBR_DL": 10e6,
        "GBR_UL": 5e6,
        "latency_ul": 10,
        "latency_dl": 10,
    },
    NETWORK_SLICE_URLLC_NAME: {
        "5QI": 1,
        "GBR_DL": 1e6,
        "GBR_UL": 0.5e6,
        "latency_ul": 0.5,
        "latency_dl": 0.5,
    },
    NETWORK_SLICE_MTC_NAME: {
        "5QI": 5,
        "GBR_DL": 0.1e6,
        "GBR_UL": 0.05e6,
        "latency_ul": 25,
        "latency_dl": 25,
    },
}
