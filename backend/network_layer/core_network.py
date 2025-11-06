import settings
import random
from .ue import UE

class CoreNetwork:
    def __init__(self, simulation_engine=None):
        self.simulation_engine = simulation_engine
        self.ue_subscription_data = settings.CORE_UE_SUBSCRIPTION_DATA.copy()
        self.active_ues = {}

    def handle_ue_authentication_and_registration(self, ue, requested_slice=None):
        slice_type, qos_profile = self.select_ue_slice_and_qos_profile(ue, requested_slice=requested_slice)
        ue.slice_type = slice_type
        ue.qos_profile = qos_profile
        ue_reg_res = {"ue": ue, "slice_type": slice_type, "qos_profile": qos_profile}
        self.active_ues[ue.ue_imsi] = ue_reg_res
        return ue_reg_res.copy()

    def select_ue_slice_and_qos_profile(self, ue, requested_slice=None):
        # If the IMSI is not in subscriptions and a requested_slice is given, allow it.
        if ue.ue_imsi not in self.ue_subscription_data:
            if requested_slice and requested_slice in settings.NETWORK_SLICES:
                self.ue_subscription_data[ue.ue_imsi] = [requested_slice]
                return requested_slice, settings.NETWORK_SLICES[requested_slice].copy()
            return None, None
        subscribed_slice_types = self.ue_subscription_data[ue.ue_imsi]
        if requested_slice and requested_slice in subscribed_slice_types:
            slice_type = requested_slice
        else:
            slice_type = random.choice(subscribed_slice_types)
        qos_profile = settings.NETWORK_SLICES[slice_type].copy()
        return slice_type, qos_profile

    # The register_new_ue method and any UE instantiation logic have been removed.
    # UE addition and random initialization should now be handled entirely in SimulationEngine.

    def handle_deregistration_request(self, ue):
        if ue.ue_imsi in self.active_ues:
            print(f"CoreNetwork: Deregistering UE {ue.ue_imsi}")
            del self.active_ues[ue.ue_imsi]
        else:
            print(f"CoreNetwork: UE {ue.ue_imsi} not found in active UEs.")
