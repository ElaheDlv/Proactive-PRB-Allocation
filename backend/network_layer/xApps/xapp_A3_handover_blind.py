from .xapp_base import xAppBase
from utils import xAppControlAction


class xAppA3HandoverBlind(xAppBase):
    """
    xApp that blindly performs handover actions upon receiving RRC measurement event A3.
    """

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = True

    def handle_rrc_meas_event_A3(self, event):
        ue = event["triggering_ue"]
        print(f"{self.xapp_id}: Received RRC measurement event A3 for UE {ue.ue_imsi}")
        print(event)

        # blindly perform handover
        return xAppControlAction(
            action_type=xAppControlAction.ACTION_TYPE_HANDOVER,
            action_data={
                "ue": ue,
                "source_cell_id": event["current_cell_id"],
                "target_cell_id": event["best_neighbour_cell_id"],
            },
        )

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: xApp is not enabled")
            return
        # subcribe events from all base stations
        for bs in self.base_station_list.values():
            bs.init_rrc_measurement_event_handler("A3", self.handle_rrc_meas_event_A3)
