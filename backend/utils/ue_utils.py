import random
import settings


class RRCMeasurementEventMonitorBase:
    def __init__(self, event_id, time_to_trigger_in_sim_steps):
        self.event_id = event_id
        self.time_to_trigger_in_sim_steps = time_to_trigger_in_sim_steps

        assert (
            self.time_to_trigger_in_sim_steps is not None
            and self.time_to_trigger_in_sim_steps > 0
        ), "Time to trigger cannot be None or less than 1"

        self.trigger_history = []

    def update_trigger_history(self, check_result):
        self.trigger_history.append(check_result)
        if len(self.trigger_history) > self.time_to_trigger_in_sim_steps:
            self.trigger_history.pop(0)

    def reset_trigger_history(self):
        self.trigger_history = []

    @property
    def is_triggered(self):
        return len(self.trigger_history) == self.time_to_trigger_in_sim_steps and all(
            check_result["triggered"] for check_result in self.trigger_history
        )

    def gen_event_report(self):
        return {
            "event_id": self.event_id,
            **self.trigger_history[-1],
        }


class RRCMeasurementEventA3Monitor(RRCMeasurementEventMonitorBase):
    def __init__(self, event_id, time_to_trigger_in_sim_steps, power_threshold):
        super().__init__(event_id, time_to_trigger_in_sim_steps)
        self.power_threshold = power_threshold

        assert self.event_id == "A3", "Only A3 event is supported for this class"
        assert self.power_threshold is not None, "Power threshold cannot be None"

    def check(self, ue, cell_signal_measurements: dict):
        current_cell = ue.current_cell
        if current_cell is None:
            return

        if len(cell_signal_measurements) <= 1:
            # no other cells to compare with
            return

        current_cell_signal_power = cell_signal_measurements.get(
            current_cell.cell_id, None
        )
        if current_cell_signal_power is None:
            return

        # remove the current_cell signal power from the the cell signal measurements
        del cell_signal_measurements[current_cell.cell_id]

        best_neighbour_cell_id = None
        best_neighbour_cell_signal_power = None
        for cell_id, signal_power in cell_signal_measurements.items():
            if (
                best_neighbour_cell_signal_power is None
                or signal_power > best_neighbour_cell_signal_power
            ):
                best_neighbour_cell_id = cell_id
                best_neighbour_cell_signal_power = signal_power

        # check if any of the other cells have a signal power greater than the current cell signal power by the power threshold
        check_result = {
            "triggering_ue": ue,
            "current_cell_id": current_cell.cell_id,
            "current_cell_signal_power": current_cell_signal_power,
            "best_neighbour_cell_id": best_neighbour_cell_id,
            "best_neighbour_cell_signal_power": best_neighbour_cell_signal_power,
            "cell_signal_measurements": cell_signal_measurements,
        }
        if (
            best_neighbour_cell_signal_power - current_cell_signal_power
            > self.power_threshold
        ):
            check_result["triggered"] = True
        else:
            check_result["triggered"] = False

        self.update_trigger_history(check_result)


def get_rrc_measurement_event_monitor(event_id, event_params):
    # only A3 event is supported for now
    if event_id == "A3":
        return RRCMeasurementEventA3Monitor(
            event_id=event_id,
            time_to_trigger_in_sim_steps=event_params["time_to_trigger_in_sim_steps"],
            power_threshold=event_params["power_threshold"],
        )
    else:
        raise ValueError(f"Unsupported rrc_measurement_event: {event_id}")


# Map SINR (in dB) to CQI
def sinr_to_cqi(sinr_db):
    thresholds = [
        -6.7,
        -4.7,
        -2.3,
        0.2,
        2.4,
        4.3,
        6.3,
        8.4,
        10.3,
        11.7,
        14.1,
        16.3,
        18.7,
        21,
        22.7,
    ]

    for i, t in enumerate(thresholds):
        if sinr_db < t:
            return i
    return 15


def get_random_ue_operational_region(step=100):
    # Choose min/max x/y as multiples of 100
    min_x = random.randint(0, (settings.NETWORK_COVERAGE_WIDTH - step) // step) * step
    min_y = random.randint(0, (settings.NETWORK_COVERAGE_HEIGHT - step) // step) * step

    # max_x/min_x at least 100m apart, at most map_size
    max_x = (
        random.randint((min_x + step) // step, settings.NETWORK_COVERAGE_WIDTH // step)
        * step
    )
    max_y = (
        random.randint((min_y + step) // step, settings.NETWORK_COVERAGE_HEIGHT // step)
        * step
    )

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
    }
