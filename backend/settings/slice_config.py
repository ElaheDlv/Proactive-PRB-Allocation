# ---------------------------
# Network Slice Configuration
# ---------------------------
NETWORK_SLICE_EMBB_NAME = "eMBB"
NETWORK_SLICE_URLLC_NAME = "URLLC"

NETWORK_SLICES = {
    NETWORK_SLICE_EMBB_NAME: {
        "5QI": 9,
        "GBR_DL": 10e6,
        "GBR_UL": 5e6,
        "latency_ul": 0.1,
        "latency_dl": 0.01,
    },
    NETWORK_SLICE_URLLC_NAME: {
        "5QI": 1,
        "GBR_DL": 1e6,
        "GBR_UL": 0.5e6,
        "latency_ul": 0.001,
        "latency_dl": 0.001,
    },
}
