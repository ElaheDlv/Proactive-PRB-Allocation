import math

# ---------------------------
# Channel Configuration
# ---------------------------
CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_LOS = "urban_macro_line_of_sight"
CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS = "urban_macro_no_line_of_sight"


def pass_loss_urban_macro_los(distance_m, frequency_ghz):
    """3GPP UMa LOS Path Loss Model (TR 38.901)"""
    if distance_m <= 0:
        return 0
    return 28 + 22 * math.log10(distance_m) + 20 * math.log10(frequency_ghz)


def path_loss_urban_macro_nlos(distance_m, frequency_ghz, h_ue=1.5):
    """3GPP UMa NLOS Path Loss Model (TR 38.901 Section 7.4.1)"""
    if distance_m <= 0 or frequency_ghz <= 0:
        raise ValueError("Distance and frequency must be positive.")

    # LOS path loss (for comparison)
    pl_los = pass_loss_urban_macro_los(distance_m, frequency_ghz)

    # NLOS calculation
    log_d = math.log10(distance_m)
    log_f = math.log10(frequency_ghz)
    nlos_pl = 13.54 + 39.08 * log_d + 20 * log_f - 0.6 * (h_ue - 1.5)

    return max(pl_los, nlos_pl)


CHANNEL_PASS_LOSS_MODEL_MAP = {
    CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_LOS: pass_loss_urban_macro_los,
    CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS: path_loss_urban_macro_nlos,
}

