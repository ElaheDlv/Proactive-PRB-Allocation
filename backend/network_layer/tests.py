import settings
import numpy as np
import matplotlib.pyplot as plt
import utils

##################################
#   Ues the plot below to tune the cell qrx_level_min against the cell radius.
##################################

test_cells = settings.RAN_BS_DEFAULT_CELLS("bs_test")

# Get the path loss model
pass_loss_model = settings.CHANNEL_PASS_LOSS_MODEL_MAP[
    settings.CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS
]

# Define distances (in meters)
distances = np.linspace(1, 6000, 500)  # From 1m to 500m

# Plot received power for each cell
plt.figure(figsize=(10, 6))
for cell in test_cells:
    received_powers = []
    for distance in distances:
        # Calculate received power
        received_power = (
            cell["transmit_power_dBm"]
            - pass_loss_model(
                distance_m=distance, frequency_ghz=cell["carrier_frequency_MHz"] / 1000
            )
            # + cell["cell_individual_offset_dBm"]
        )
        received_powers.append(received_power)

    # Plot the results
    plt.plot(
        distances,
        received_powers,
        label=f"{cell['cell_id']} ({cell['carrier_frequency_MHz']} MHz)",
    )

# Add labels, legend, and title
plt.xlabel("Distance (m)")
plt.ylabel("Received Power (dBm)")
plt.title("Received Power vs Distance for Test Cells")
plt.legend()
plt.grid()
plt.show()


# Constants for noise calculation
k = 1.38e-23  # Boltzmann constant
temperature_k = settings.UE_TEMPERATURE_K  # User equipment temperature in Kelvin

# Plot SINR for each cell
plt.figure(figsize=(10, 6))
for cell in test_cells:
    sinrs = []
    for distance in distances:
        # Calculate received power
        received_power_dBm = (
            cell["transmit_power_dBm"]
            - pass_loss_model(
                distance_m=distance, frequency_ghz=cell["carrier_frequency_MHz"] / 1000
            )
            # + cell["cell_individual_offset_dBm"]
        )
        received_power_w = utils.dbm_to_watts(received_power_dBm)  # Convert dBm to Watts

        # Calculate noise power
        noise_power_w = k * temperature_k * cell["bandwidth_Hz"]

        # Calculate interference power (assuming no other cells for simplicity)
        interference_power_w = 0  # Adjust if interference from other cells is considered

        # Calculate SINR
        sinr = received_power_w / (interference_power_w + noise_power_w)
        sinr_dB = 10 * np.log10(sinr)  # Convert SINR to dB
        sinrs.append(sinr_dB)

    # Plot the SINR
    plt.plot(
        distances,
        sinrs,
        label=f"{cell['cell_id']} ({cell['carrier_frequency_MHz']} MHz)",
    )

# Add labels, legend, and title
plt.xlabel("Distance (m)")
plt.ylabel("SINR (dB)")
plt.title("SINR vs Distance for Test Cells")
plt.legend()
plt.grid()
plt.show()