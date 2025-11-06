from typing import Any
from agents import function_tool
from pydantic import BaseModel
from pydantic import Field
from settings.slice_config import NETWORK_SLICE_EMBB_NAME, NETWORK_SLICE_MTC_NAME, NETWORK_SLICE_URLLC_NAME
from network_layer.simulation_engine import SimulationEngine

class UEDetails:
    def __init__(self, imsi: str, network_slices: list[str]):
        self.IMSI = imsi
        self.NETWORK_SLICES = network_slices

    def __repr__(self):
        return f"UEDetails(imsi={self.IMSI}, NETWORK_SLICE={self.NETWORK_SLICES})"
    
    def to_dict(self):
        return {
            'IMSI': self.IMSI,
            'NETWORK_SLICES': self.NETWORK_SLICES
        }

@function_tool
def add_ue(imsi: str, network_slices: list[str]):
    """
    Dynamically adds a UE with the given IMSI and network_slices to the running simulation.
    Args:
        imsi (str): The IMSI of the UE, it should be like IMSI_<number>.
        network_slices (list[str]): A list of network slices that the UE is allowed to use.
                                    It can be at least one of [eMBB, URLLC, mMTC].
    Returns:
        str: Status message.
    """
    sim_engine = SimulationEngine()
    success = sim_engine.register_ue(imsi, network_slices)
    if success:
        return f"UE {imsi} was successfully registered."
    else:
        return f"Failed to register UE {imsi}."

@function_tool
def remove_ue(imsi: str):
    """
    Deregisters and removes a UE from the running simulation.

    Args:
        imsi (str): The IMSI of the UE.
    Returns:
        str: Status message.
    """
    sim_engine = SimulationEngine()
    success = sim_engine.deregister_ue(imsi)
    if success:
        return f"UE {imsi} was successfully deregistered and removed."
    else:
        return f"Failed to deregister or find UE {imsi}."

@function_tool
def get_available_ue_description() -> str:
    """
    Provides a description of available UEs that are currently registered in the core network.
    Returns:
        str: A summary of registered UEs (first and last IMSI), or absence message.
    """
    sim_engine = SimulationEngine()
    core_network = sim_engine.core_network
    imsies = list(core_network.active_ues.keys())
    if not imsies:
        return "No UEs available."
    first_ue = imsies[0]
    last_ue = imsies[-1]
    if first_ue == last_ue:
        return f"Available UEs: There is only one UE with imsi: {first_ue}"
    else:
        return f"Available UEs: First UE imsi is {first_ue}, Last UE imsi is {last_ue}."

class GetUE(BaseModel):
    IMSI: str = Field(description = "The name of the subscription always starts with IMSI_")
    NETWORK_SLICES: list[str] = Field(description="The allocated network slices to the subscription should be atleast one of [eMBB, uRLLC, mMTC]")

@function_tool
def get_ues(slices: list[str] = [NETWORK_SLICE_EMBB_NAME, NETWORK_SLICE_URLLC_NAME, NETWORK_SLICE_MTC_NAME]) -> list[GetUE]:
    """
    Retrieve all registered User Equipments (UEs) from the core network along with their IMSIs and network slice subscriptions.

    Args:
        slices (list[str], optional): A list of network slices to filter UEs by. 
            Only UEs subscribed to at least one of these slices will be returned.
            If the list is empty or contains all possible slices, all UEs will be returned.

    Returns:
        list[GetUE]: A list of UEs and their associated subscriptions, with each UE represented as:
            - IMSI: The unique identifier of the UE (e.g., IMSI_101).
            - NETWORK_SLICES: The list of network slices this UE is currently subscribed to.

    Example:
        get_ues() 
            → Returns all UEs and their subscriptions.
        get_ues(slices=['eMBB'])
            → Returns only UEs that are subscribed to the eMBB slice.

    Notes:
        - If no UEs are registered, returns an empty list.
        - The slices parameter allows selective retrieval for scenarios where only UEs with access to certain slices are needed.
    """
    print(f"LOG: slices: {slices}")
    sim_engine = SimulationEngine()
    core_network = sim_engine.core_network
    result: list[GetUE] = []
    for ue_imsi, reg_info in core_network.active_ues.items():
        print(f"LOG: {ue_imsi}")
        subscribed_slices = core_network.ue_subscription_data.get(ue_imsi, [])
        if not any(slice in subscribed_slices for slice in slices):
            continue
        result.append(GetUE(IMSI = ue_imsi, NETWORK_SLICES = subscribed_slices))
    return result

@function_tool
def update_ue_subscription(imsi: str, new_slices: list[str]):
    """
    Updates the network slices for an existing UE (by IMSI).
    Args:
        imsi (str): The IMSI of the UE.
        new_slices (list[str]): The updated list of allowed network slices.
    Returns:
        str: Status message.
    """
    sim_engine = SimulationEngine()
    core_network = sim_engine.core_network

    if not isinstance(new_slices, list) or not new_slices:
        return f"Error: new_slices must be a non-empty list."

    # Check if UE is present in the network's subscription registry
    if imsi not in core_network.ue_subscription_data:
        return f"UE {imsi} not found among registered UEs."

    # Update the slice subscription in the core network
    core_network.ue_subscription_data[imsi] = new_slices

    # If UE is connected/active, optionally update its registration (slice assignment).
    if imsi in core_network.active_ues:
        ue = core_network.active_ues[imsi]["ue"]
        # Re-authenticate/assign new slice (pick first in list as default, or random)
        main_slice = new_slices[0]
        core_network.handle_ue_authentication_and_registration(ue, requested_slice=main_slice)

    return f"Subscription for UE {imsi} has been updated. New slices: {new_slices}"