import math
import settings


def dist_between(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Convert dBm to linear scale (Watts)
def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)


# Convert Watts to dBm
def watts_to_dbm(watts):
    return 10 * math.log10(watts) + 30


def estimate_throughput(ue_modulation_order, ue_code_rate, ue_dl_prb):
    # only downlink bitrate is supported for now.

    # Bits per RE (Resource Element)
    bits_per_symbol = ue_modulation_order * ue_code_rate / 1024

    # Bits per PRB per slot
    bits_per_prb_per_slot = (
        bits_per_symbol * settings.RAN_RESOURCE_ELEMENTS_PER_PRB_PER_SLOT
    )

    # DL & UL TBS estimates (bits per slot)
    estimated_tbs_dl = ue_dl_prb * bits_per_prb_per_slot
    # estimated_tbs_ul = self.prb_ue_allocation_dict[ue.ue_imsi]["uplink"] * bits_per_prb_per_slot

    # Throughput = bits per second
    dl_throughput = estimated_tbs_dl / settings.RAN_SLOT_DURATION
    # ul_throughput = estimated_tbs_ul / settings.RAN_SLOT_DURATION

    return dl_throughput
