import json
from ..knowledge_entry import knowledge_entry
from ..tags import KnowledgeTag
from ..relationships import KnowledgeRelationship
from settings import (
    NETWORK_SLICES,
    NETWORK_SLICE_EMBB_NAME,
    NETWORK_SLICE_URLLC_NAME,
    NETWORK_SLICE_MTC_NAME,
)

import inspect
from network_layer.ue import UE


SUPPORTED_UE_ATTRIBUTES = [
    "ue_imsi",
    "position_x",
    "position_y",
    "target_x",
    "target_y",
    "speed_mps",
    "time_remaining",
    "slice_type",
    "qos_profile",
    "connected",
    "downlink_bitrate",
    # "downlink_latency",
    "downlink_sinr",
    "downlink_cqi",
    "downlink_mcs_index",
    "downlink_mcs_data",
    # "uplink_bitrate",
    # "uplink_latency",
    # "uplink_transmit_power_dBm",
    # "serving_cell_history",
    "current_cell",
    # rrc_measurement_event_monitors
    # downlink_received_power_dBm_dict
]

SUPPORTED_UE_METHODS = [
    "power_up",
    "monitor_signal_strength",
    "cell_selection_and_camping",
    "authenticate_and_register",
    "set_downlink_bitrate",
    "set_downlink_mcs_index",
    "set_downlink_mcs_data",
    "set_downlink_sinr",
    "set_downlink_cqi",
    "move_towards_target",
    "setup_rrc_measurement_event_monitors",
    "check_rrc_meas_events_to_monitor",
    "execute_handover",
    "set_current_cell",
    "deregister",
    "calculate_SINR_and_CQI",
    "step",
]


@knowledge_entry(
    key="/docs/user_equipments",
    tags=[KnowledgeTag.UE, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def ue_knowledge_help(sim, knowledge_router, query_key, params):
    return (
        "📘 **Welcome to the User Equipment (UE) Knowledge Base**\n\n"
        "You can query live data and explanations for User Equipments (UEs) in the simulation.\n\n"
        "### Available Endpoints:\n"
        "- **List all UEs (identifiers only)**: `/user_equipments`\n"
        "- **Get all attributes values for a specific UE**: `/user_equipments/{ue_imsi}`\n"
        "- **Get a specific attribute value of a specific UE**: `/user_equipments/{ue_imsi}/attributes/{attribute_name}`\n"
        "- **Explain what an attribute of UE means**: `/docs/user_equipments/attributes/{attribute_name}`\n"
        "- **Explain what a method in UE class does**: `/docs/user_equipments/methods/{method_name}`\n"
        "### Supported UE Attributes:\n"
        f"{', '.join(SUPPORTED_UE_ATTRIBUTES)}\n\n"
        "### Supported UE Methods:\n"
        f"{', '.join(SUPPORTED_UE_METHODS)}\n\n"
    )


# ------------------------------------------
#     GET /user_equipments
#       → List all UEs (representations) in the simulation
# ------------------------------------------
@knowledge_entry(
    key="/user_equipments",
    tags=[KnowledgeTag.UE],
    related=[],
)
def get_ue_repr_list(sim, knowledge_router, query_key, params):
    list_of_ue_repr = [repr(ue) for ue in sim.ue_list.values()]
    return (
        f"Currently there are {len(list_of_ue_repr)} UEs in the simulation:\n"
        f"{"\n".join(list_of_ue_repr)}\n"
    )


# ------------------------------------------
#     GET /user_equipments/{ue_imsi}
#       → List all attributes values for the given UE
# ------------------------------------------
@knowledge_entry(
    key="/user_equipments/{ue_imsi}",
    tags=[KnowledgeTag.UE],
    related=[],
)
def get_ue_attributes(sim, knowledge_router, query_key, params):
    ue_imsi = params["ue_imsi"]
    ue = sim.ue_list.get(ue_imsi)
    if ue is None:
        return f"UE with IMSI: {ue_imsi} not found."
    response = f"Attributes of UE {ue_imsi}:\n"
    for attr in SUPPORTED_UE_ATTRIBUTES:
        value = getattr(ue, attr, None)
        response += f"- {attr}: {repr(value)}\n"

    return response


# ------------------------------------------
#     GET /user_equipments/{ue_imsi}/attributes/{attribute_name}
#       → Get a specific attribute value for the given UE
# ------------------------------------------
@knowledge_entry(
    key="/user_equipments/{ue_imsi}/attributes/{attribute_name}",
    tags=[KnowledgeTag.UE],
    related=[],
)
def get_ue_attribute_value(sim, knowledge_router, query_key, params):
    ue_imsi = params["ue_imsi"]
    attribute_name = params["attribute_name"]
    ue = sim.ue_list.get(ue_imsi)
    if ue is None:
        return f"UE with IMSI: {ue_imsi} not found."
    if attribute_name not in SUPPORTED_UE_ATTRIBUTES:
        return f"Attribute '{attribute_name}' is not supported."
    value = getattr(ue, attribute_name, None)
    return f"Value of {attribute_name} for UE {ue_imsi}: {repr(value)}"


# ------------------------------------------
#     GET /user_equipments/attributes/{attribute_name}
#       → Explain what the attribute means
# ------------------------------------------
@knowledge_entry(
    "/docs/user_equipments/attributes/ue_imsi",
    tags=[KnowledgeTag.UE, KnowledgeTag.ID],
    related=[],
)
def ue_imsi_explainer(sim, knowledge_router, query_key, params):
    return (
        "The UE's IMSI (International Mobile Subscriber Identity) is a unique identifier "
        "assigned to the UE. Examples in this simulated network are IMSI_1, IMSI_32, etc."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/position_x",
    tags=[KnowledgeTag.UE, KnowledgeTag.LOCATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        ),
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/position_y",
        ),
    ],
)
def ue_position_x_explainer(sim, knowledge_router, query_key, params):
    return f"The UE's x-coordinate (unit: m) in the simulated area."


@knowledge_entry(
    "/docs/user_equipments/attributes/position_y",
    tags=[KnowledgeTag.UE, KnowledgeTag.LOCATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        ),
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/position_x",
        ),
    ],
)
def ue_position_y_explainer(sim, knowledge_router, query_key, params):
    return f"The UE's y-coordinate (unit: m) in the simulated area."


@knowledge_entry(
    "/docs/user_equipments/attributes/target_x",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        )
    ],
)
def ue_target_x_explainer(sim, knowledge_router, query_key, params):
    return f"The UE's target x-coordinate (unit: m) in the simulated area."


@knowledge_entry(
    "/docs/user_equipments/attributes/target_y",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        )
    ],
)
def ue_target_y_explainer(sim, knowledge_router, query_key, params):
    return f"The UE's target y-coordinate (unit: m) in the simulated area."


@knowledge_entry(
    "/docs/user_equipments/attributes/speed_mps",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        )
    ],
)
def ue_speed_mps_explainer(sim, knowledge_router, query_key, params):
    return """The UE's speed (unit: m/s) in the simulated area. 
The UE will move towards its target position at this speed."""


@knowledge_entry(
    "/docs/user_equipments/attributes/time_remaining",
    tags=[KnowledgeTag.UE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/user_equipments/methods/step",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/step",
        ),
    ],
)
def ue_time_remaining_explainer(sim, knowledge_router, query_key, params):
    return (
        "The number of simulation steps remaining for the UE, "
        "which is decremented by 1 every simulation step. "
        "When it reaches 0, the UE will deregister itself from the newtork "
        "and removed from the simulation."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/slice_type",
    tags=[KnowledgeTag.UE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/qos_profile",
        ),
    ],
)
def ue_slice_type_explainer(sim, knowledge_router, query_key, params):
    return f"""
The UE's slice type is randomly selected during the authentication and registration process, 
which can be one of the following:
* eMBB (Enhanced Mobile Broadband): Designed to provide high data rates and capacity for applications like video streaming.
    corresponding QoS profile: {json.dumps(NETWORK_SLICES[NETWORK_SLICE_EMBB_NAME])}
* URLLC (Ultra-Reliable Low Latency Communication): Designed for applications requiring ultra-reliable and low-latency communication, such as autonomous driving.
    corresponding QoS profile: {json.dumps(NETWORK_SLICES[NETWORK_SLICE_URLLC_NAME])}
* mMTC (Massive Machine Type Communication): Designed for applications with a large number of low-power devices, such as IoT sensors.
    corresponding QoS profile: {json.dumps(NETWORK_SLICES[NETWORK_SLICE_MTC_NAME])}

When the UE picks up a slice type, it will also get the corresponding QoS profile 
(query /user_equipments/attribute/qos_profile for more details).
"""


@knowledge_entry(
    "/docs/user_equipments/attributes/qos_profile",
    tags=[KnowledgeTag.UE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/slice_type",
        ),
    ],
)
def ue_qos_profile_explainer(sim, knowledge_router, query_key, params):
    # v_5qi = f"5QI: {slice_qos_profile.get("5QI")}"
    # v_gbr_dl = f"GBR_DL: {slice_qos_profile.get("GBR_DL")}"
    # v_gbr_ul = f"GBR_UL: {slice_qos_profile.get("GBR_UL")}"
    # v_latency_dl = f"latency_dl: {slice_qos_profile.get("latency_dl")}"
    # v_latency_ul = f"latency_ul: {slice_qos_profile.get("latency_ul")}"

    return """The UE's QoS (Quality of Service) profile is a set of parameters that define how the network should treat the UE's traffic:
    * 5QI is a numerical identifier (e.g., 1, 9, 66, etc.) that maps to a specific QoS profile — a predefined set of parameters that define how the network should treat a particular type of traffic. It simplifies QoS management by bundling parameters under a single value.
    * GBR_DL (Guaranteed Bit Rate Downlink) is the minimum guaranteed data rate for downlink traffic. It ensures that the user equipment (UE) receives a certain amount of bandwidth for its downlink communication. 
    * GBR_UL (Guaranteed Bit Rate Uplink) is the minimum guaranteed data rate for uplink traffic. It ensures that the UE can send a certain amount of data to the network without being throttled.
    * latency_ul (Uplink Latency) is the maximum time it takes for a packet to travel from the UE to the network. 
    * latency_dl (Downlink Latency) is the maximum time it takes for a packet to travel from the network to the UE. 

Note that the actual bitrate and latency (get_knowledge_value: /user_equipments/attribute/{ue_imsi}/downlink_bitrate, /user_equipments/attribute/{ue_imsi}/uplink_bitrate, /user_equipments/attribute/{ue_imsi}/uplink_latency, /user_equipments/attribute/{ue_imsi}/downlink_latency) may vary based on network conditions and other factors.
The values provided in the QoS profile are guarantees that the network aims to meet, but they are not absolute limits."""


@knowledge_entry(
    "/docs/user_equipments/attributes/connected",
    tags=[KnowledgeTag.UE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/deregister",
        ),
    ],
)
def ue_connected_explainer(sim, knowledge_router, query_key, params):
    return "A boolean representing whether the UE is connected or not to the network (served by a cell and registered with the core)."


@knowledge_entry(
    "/docs/user_equipments/attributes/downlink_bitrate",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/set_downlink_bitrate",
        ),
    ],
)
def ue_downlink_bitrate_explainer(sim, knowledge_router, query_key, params):
    return (
        "The UE's actual downlink bitrate (unit: bps), calculated by the serving cell."
        " Comparing this value to the UE's QoS profile (get_knowledge_value: /user_equipments/attribute/{ue_imsi}/qos_profile) "
        "can help determine if the UE is receiving the expected service quality."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/downlink_sinr",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DEPENDS_ON,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (
            KnowledgeRelationship.DEPENDS_ON,
            "/docs/cell/attribute/transmit_power_dBm",
        ),
        (
            KnowledgeRelationship.AFFECTS,
            "/docs/user_equipments/attributes/downlink_cqi",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/calculate_SINR_and_CQI",
        ),
    ],
)
def ue_downlink_sinr_explainer(sim, knowledge_router, query_key, params):
    return (
        "The downlink SINR (Signal-to-Interference-plus-Noise Ratio) attribute quantifies the quality of the radio signal received by the UE from its serving cell. "
        "A higher SINR indicates better signal quality, which enables higher data rates and more reliable communication. "
        "Typical SINR value ranges are interpreted as follows: values above 20 dB are considered excellent, 13–20 dB are good, 0–13 dB are acceptable, -10–0 dB are poor, and values below -10 dB indicate very poor signal quality. "
        "SINR directly impacts the selection of modulation and coding schemes and is a key factor in link adaptation and handover decisions."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/downlink_cqi",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/user_equipments/attributes/downlink_sinr",
        ),
        (
            KnowledgeRelationship.AFFECTS,
            "/docs/user_equipments/attributes/downlink_mcs_index",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/calculate_SINR_and_CQI",
        ),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/cell/methods/select_ue_mcs",
        ),
    ],
)
def ue_downlink_cqi_explainer(sim, knowledge_router, query_key, params):
    return (
        "The downlink CQI (Channel Quality Indicator) attribute is a metric reported by the UE to indicate the perceived quality of the downlink channel. "
        "CQI is derived from the measured SINR and is used by the network to select the most appropriate modulation and coding scheme (MCS) for data transmission. "
        "Higher CQI values correspond to better channel conditions, allowing the network to use more efficient (higher-rate) MCS settings, which increases throughput. "
        "Lower CQI values indicate poorer channel conditions, prompting the network to use more robust but less efficient MCS settings to maintain reliable communication. "
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/downlink_mcs_index",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/user_equipments/attributes/downlink_cqi",
        ),
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/downlink_mcs_data",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/cell/methods/select_ue_mcs",
        ),
    ],
)
def ue_downlink_mcs_index_explainer(sim, knowledge_router, query_key, params):
    return (
        "The downlink MCS (Modulation and Coding Scheme) index attribute specifies which modulation and coding scheme is currently assigned to the UE for downlink data transmission. "
        "The MCS index determines both the modulation order (e.g., QPSK, 16QAM, 64QAM) and the coding rate, directly impacting the data rate and reliability of the communication link. "
        "Higher MCS indices correspond to higher data rates but require better channel conditions, as indicated by the UE's CQI. "
        "If the MCS index is set to -1, it typically means that the UE is not currently assigned a valid MCS, possibly due to poor channel conditions or lack of resource allocation."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/downlink_mcs_data",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS],
    related=[
        (
            KnowledgeRelationship.ASSOCIATED_WITH,
            "/docs/user_equipments/attributes/downlink_mcs_index",
        ),
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/cell/methods/select_ue_mcs",
        ),
        (
            KnowledgeRelationship.DERIVED_FROM,
            "/docs/user_equipments/attributes/downlink_cqi",
        ),
        (
            KnowledgeRelationship.USED_BY_METHOD,
            "/docs/cell/methods/estimate_ue_bitrate_and_latency",
        ),
    ],
)
def ue_downlink_mcs_data_explainer(sim, knowledge_router, query_key, params):
    return (
        "The downlink MCS (Modulation and Coding Scheme) data attribute contains detailed parameters associated with the currently assigned MCS index. "
        "These parameters typically include the modulation order (number of bits per symbol), the target code rate (fraction of useful data in the transmitted bits), and the spectral efficiency (data rate per unit bandwidth). "
        "Higher modulation orders and code rates enable higher data rates but require better channel conditions. "
        "The MCS data is selected based on the UE's reported CQI and is essential for determining the actual throughput achievable by the UE."
    )


@knowledge_entry(
    "/docs/user_equipments/attributes/current_cell",
    tags=[KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.SET_BY_METHOD,
            "/docs/user_equipments/methods/cell_selection_and_camping",
        )
    ],
)
def ue_current_cell_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `current_cell` attribute indicates which cell the UE (User Equipment) is currently connected to in the network. "
        "The `current_cell` can change over time due to handover procedures, "
        "ensuring the UE maintains optimal connectivity as it moves or as radio conditions change. "
    )


# ------------------------------------------
#    GET /user_equipments/methods/{method_name}
#      → Explain what the method does
# ------------------------------------------
@knowledge_entry(
    key="/docs/user_equipments/methods/power_up",
    tags=[KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/cell_selection_and_camping",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
    ],
)
def ue_power_up_explainer(sim, knowledge_router, query_key, params):
    # Fetch the source code from the UE class
    try:
        method_source = inspect.getsource(getattr(UE, "power_up"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `power_up` method is responsible for initializing the UE's connection to the network. "
        "This process involves several steps:\n\n"
        "1. **Signal Strength Monitoring**: The UE scans for available cells and measures their signal strength. "
        "Cells are ranked based on their received power and frequency priority. (method query key: /user_equipments/method/monitor_signal_strength).\n\n"
        "2. **Cell Selection and Camping**: The UE selects the most suitable cell based on the measured signal strength "
        "and camps on it. This step ensures that the UE is connected to the cell with the best signal quality. (method query key: /user_equipments/method/cell_selection_and_camping)\n\n"
        "3. **Authentication and Registration**: The UE performs authentication and registration with the selected cell's base station. "
        "During this step, the UE is assigned a network slice type (e.g., eMBB, URLLC, or mMTC) and a corresponding QoS profile. (method query key: /user_equipments/method/authenticate_and_register).\n\n"
        "\nOnce the above steps are successfully completed, the UE is marked as connected to the network."
    )
    explanation_text += (
        " If any of these steps fail (e.g., no cells are detected, or authentication fails), "
        "the UE will not be added to the simulation."
    )
    return code_block + explanation_text


@knowledge_entry(
    key="/docs/user_equipments/methods/monitor_signal_strength",
    tags=[KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/calculate_SINR_and_CQI",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_sinr",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_cqi",
        ),
    ],
)
def ue_monitor_signal_strength_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "monitor_signal_strength"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `monitor_signal_strength` method allows the UE to scan for available cells "
        "and measure their downlink signal strength. This process involves:\n\n"
        "1. **Path Loss Calculation**: The UE calculates the received power from each cell "
        "based on the cell's transmit power, distance, and the path loss model.\n\n"
        "2. **Filtering Detected Cells**: Only cells with received power above the detection threshold "
        "and meeting the minimum quality requirements are considered.\n\n"
        "3. **SINR and CQI Calculation**: calls another method `/user_equipments/method/calculate_SINR_and_CQI` where "
        "the UE calculates the Signal-to-Interference-plus-Noise Ratio (SINR) "
        "and derives the Channel Quality Indicator (CQI) for the current serving cell (/user_equipments/attribute/current_cell) if applicable.\n\n"
        "This method is critical for determining the UE's connectivity and channel quality, "
        "which influence procedures such as cell selection, resource allocation and RRC event monitoring."
    )
    return code_block + explanation_text


@knowledge_entry(
    key="/docs/user_equipments/methods/cell_selection_and_camping",
    tags=[KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.DEPENDS_ON,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/current_cell",
        ),
    ],
)
def ue_cell_selection_and_camping_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "cell_selection_and_camping"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `cell_selection_and_camping` method enables the UE to select the most suitable cell "
        "to connect to. This process involves:\n\n"
        "1. **Cell Ranking**: The UE ranks detected cells based on their received power, "
        "frequency priority, and cell-specific offsets (CIO).\n\n"
        "2. **Cell Selection**: The UE selects the cell with the highest rank that meets the minimum quality requirements.\n\n"
        "3. **Camping**: The UE camps on the selected cell, marking it as the current serving cell.\n\n"
        "This method ensures that the UE connects to the best available cell, "
        "which is essential for reliable communication and efficient resource utilization."
    )
    return code_block + explanation_text


@knowledge_entry(
    key="/docs/user_equipments/methods/authenticate_and_register",
    tags=[KnowledgeTag.UE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/slice_type",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/qos_profile",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/setup_rrc_measurement_event_monitors",
        ),
    ],
)
def ue_authenticate_and_register_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "authenticate_and_register"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `authenticate_and_register` method allows the UE to authenticate with the network "
        "and register with its serving cell. This process involves:\n\n"
        f"1. **Slice Type Assignment**: The UE randomly selects one of the slice types ({', '.join(NETWORK_SLICES.keys())}).\n\n"
        "2. **QoS Profile Assignment**: Based on the selected slice type, the UE gets the associated QoS Profile, "
        "defining parameters like 5QI, guaranteed uplink/downlink bit rate (GBR), and uplink/downlink latency.\n\n"
        "3. **Setup RRC Event Monitor**: Calls method `/user_equipments/method/setup_rrc_measurement_event_monitors` to set up RRC (Radio Resource Control) event monitors to track signal quality and other parameters.\n\n"
        "This method is crucial for establishing the UE's identity and ensuring it receives "
        "the appropriate network resources and quality of service."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/setup_rrc_measurement_event_monitors",
    tags=[KnowledgeTag.UE, KnowledgeTag.CODE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/rrc_measurement_event_monitors",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/authenticate_and_register",
        ),
    ],
)
def ue_setup_rrc_measurement_event_monitors_explainer(
    sim, knowledge_router, query_key, params
):
    try:
        method_source = inspect.getsource(
            getattr(UE, "setup_rrc_measurement_event_monitors")
        )
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `setup_rrc_measurement_event_monitors` method initializes the UE's monitoring of RRC (Radio Resource Control) measurement events. "
        "These events are used to track radio conditions and trigger mobility procedures such as handover.\n\n"
        "Detailed behavior:\n"
        "1. **Input**: The method takes a list of event configuration dictionaries, each specifying an event type (e.g., 'A3'), "
        "a time-to-trigger (in simulation steps), and event-specific parameters (such as power thresholds).\n"
        "2. **Monitor Creation**: For each event configuration, the method creates an event monitor object (e.g., `RRCMeasurementEventA3Monitor`) "
        "using a factory function (such as `get_rrc_measurement_event_monitor`). Each monitor is responsible for tracking whether its event's triggering conditions are met over time.\n"
        "3. **Assignment**: The created monitors are stored in the UE's `rrc_measurement_event_monitors` attribute, replacing any previous monitors.\n"
        "4. **Usage**: During each simulation step, the UE calls `check_rrc_meas_events_to_monitor`, which iterates through these monitors, "
        "checks their triggering conditions (e.g., whether a neighboring cell's signal is sufficiently stronger than the current cell), "
        "and triggers reports or handover procedures if necessary.\n\n"
        "This method is essential for enabling dynamic and standards-compliant mobility management in the simulation, "
        "allowing the UE to autonomously detect when radio conditions warrant a handover or other RRC event-driven actions."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/check_rrc_meas_events_to_monitor",
    tags=[KnowledgeTag.UE, KnowledgeTag.CODE, KnowledgeTag.SIMULATION],
    related=[
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/rrc_measurement_event_monitors",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/user_equipments/methods/step"),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/base_station/methods/receive_ue_rrc_meas_events",
        ),
    ],
)
def ue_check_rrc_meas_events_to_monitor_explainer(
    sim, knowledge_router, query_key, params
):
    try:
        method_source = inspect.getsource(
            getattr(UE, "check_rrc_meas_events_to_monitor")
        )
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `check_rrc_meas_events_to_monitor` method is responsible for evaluating all configured RRC (Radio Resource Control) measurement event monitors for the UE at each simulation step.\n\n"
        "Detailed behavior:\n"
        "1. **Signal Map Construction**: The method first constructs a mapping of all detected cells' IDs to their received power (with CIO adjustment) from the UE's `downlink_received_power_dBm_dict` attribute. This map represents the latest signal measurements for all visible cells.\n"
        "2. **Event Monitor Evaluation**: For each RRC measurement event monitor in the UE's `rrc_measurement_event_monitors` list (typically created by `setup_rrc_measurement_event_monitors`), the method calls the monitor's `check` function, passing the UE instance and a copy of the signal map. Each monitor evaluates whether its specific event's triggering conditions are met (e.g., for Event A3: whether a neighbor cell's signal exceeds the serving cell's by a threshold for a required duration).\n"
        "3. **Trigger History Update**: Each monitor maintains a trigger history buffer (of length equal to its time-to-trigger parameter). The monitor updates this buffer with the result of each check, tracking whether the event's condition has been continuously satisfied.\n"
        "4. **Event Triggering**: If a monitor's `is_triggered` property becomes True (i.e., the event's condition has been met for the required number of consecutive steps), the method generates an event report (via the monitor's `gen_event_report` method) and sends it to the current base station by calling `self.current_bs.receive_ue_rrc_meas_events(event_report)`. This may initiate procedures such as handover.\n"
        "5. **Reporting**: The method prints diagnostic messages to the console when an event is triggered, including the event type and the generated report.\n\n"
        "This method is essential for enabling autonomous, standards-compliant mobility management in the simulation. By continuously monitoring radio conditions and evaluating RRC measurement events, the UE can detect when handover or other mobility actions are needed, ensuring robust connectivity as the UE moves or as network conditions change."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_downlink_bitrate",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/cell/methods/estimate_ue_bitrate_and_latency",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_bitrate",
        ),
    ],
)
def ue_set_downlink_bitrate_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_downlink_bitrate"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_downlink_bitrate` method is a setter used to update the UE's downlink bitrate attribute. "
        "This method does not perform any calculations itself; it simply assigns the provided bitrate value to the UE."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_downlink_mcs_index",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/cell/methods/select_ue_mcs"),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_mcs_index",
        ),
    ],
)
def ue_set_downlink_mcs_index_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_downlink_mcs_index"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_downlink_mcs_index` method is a setter for the UE's downlink MCS (Modulation and Coding Scheme) index. "
        "This method simply updates the UE's internal record of which MCS index is currently assigned."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_downlink_mcs_data",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/cell/methods/select_ue_mcs"),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_mcs_data",
        ),
    ],
)
def ue_set_downlink_mcs_data_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_downlink_mcs_data"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_downlink_mcs_data` method is a setter for the UE's downlink MCS data, "
        "which includes parameters such as modulation order, target code rate, and spectral efficiency. "
        "This method does not compute the MCS data itself, but stores the data structure provided by the cell, "
        "enabling the UE to reference its current transmission parameters."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_downlink_sinr",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_sinr",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/calculate_SINR_and_CQI",
        ),
    ],
)
def ue_set_downlink_sinr_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_downlink_sinr"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_downlink_sinr` method is a setter for the UE's downlink SINR (Signal-to-Interference-plus-Noise Ratio) attribute. "
        "It is typically called by the `calculate_SINR_and_CQI` method after the SINR value has been computed based on the received power from the serving cell, "
        "interference from neighboring cells, and thermal noise. "
        "This method does not perform any calculations itself; it simply updates the UE's internal record of the current downlink SINR value."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_downlink_cqi",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_cqi",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/calculate_SINR_and_CQI",
        ),
    ],
)
def ue_set_downlink_cqi_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_downlink_cqi"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_downlink_cqi` method is a setter for the UE's downlink CQI (Channel Quality Indicator) attribute. "
        "It is usually invoked by the `calculate_SINR_and_CQI` method after the CQI value has been derived from the current SINR measurement. "
        "This method does not compute the CQI itself; it simply stores the provided CQI value in the UE's state. "
        "The CQI value is a key input for the network's link adaptation algorithms, influencing the selection of modulation and coding schemes (MCS) "
        "and ultimately determining the data rate and reliability of the UE's downlink connection."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/move_towards_target",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/position_x",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/position_y",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/target_x",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/target_y",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/speed_mps",
        ),
    ],
)
def ue_move_towards_target_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "move_towards_target"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `move_towards_target` method updates the UE's position by moving it towards its target coordinates (`target_x`, `target_y`) "
        "based on its current speed (`speed_mps`) and the elapsed simulation time (`delta_time`).\n\n"
        "Detailed behavior:\n"
        "1. **Distance Calculation**: The method first calculates the straight-line (Euclidean) distance between the UE's current position (`position_x`, `position_y`) and its target position.\n"
        "2. **Movement Determination**: It computes the maximum distance the UE can move in this simulation step as `speed_mps * delta_time`.\n"
        "3. **Position Update**:\n"
        "   - If the target is within reach (i.e., the distance to the target is less than or equal to the maximum move distance), the UE's position is set directly to the target coordinates.\n"
        "   - Otherwise, the UE moves along the straight line towards the target by the maximum allowed distance, updating both `position_x` and `position_y` proportionally.\n"
        "   - The new position is rounded to the nearest integer to reflect discrete simulation steps.\n"
        "4. **No Movement**: If the UE is already at the target, its position remains unchanged."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/execute_handover",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_current_cell",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/current_cell",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/serving_cell_history",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_sinr",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_cqi",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_mcs_index",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_mcs_data",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_bitrate",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_latency",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/uplink_bitrate",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/uplink_latency",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/uplink_transmit_power_dBm",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/rrc_measurement_event_monitors",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/base_station/methods/execute_handover",
        ),
    ],
)
def ue_execute_handover_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "execute_handover"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `execute_handover` method enables the UE (User Equipment) to switch its connection from the current serving cell to a new target cell. "
        "This is a critical mobility management procedure in cellular networks, ensuring seamless connectivity as the UE moves or as radio conditions change.\n\n"
        "**Detailed behavior:**\n"
        "1. **Reset Radio State:** The method clears the UE's downlink received power measurements (`downlink_received_power_dBm_dict`), resets the downlink SINR (`downlink_sinr`), CQI (`downlink_cqi`), MCS index (`downlink_mcs_index`), MCS data (`downlink_mcs_data`), downlink bitrate (`downlink_bitrate`), and downlink latency (`downlink_latency`). "
        "It also resets uplink bitrate (`uplink_bitrate`), uplink latency (`uplink_latency`), and uplink transmit power (`uplink_transmit_power_dBm`). This ensures that all radio parameters are re-initialized for the new cell context.\n"
        "2. **Update Serving Cell:** The method calls `set_current_cell(target_cell)`, which updates the UE's `current_cell` attribute to the new cell and appends the cell's ID to the `serving_cell_history` list. This maintains a record of the UE's cell association over time.\n"
        "3. **Reset RRC Event Monitors:** For each configured RRC (Radio Resource Control) measurement event monitor in `rrc_measurement_event_monitors`, the method calls `reset_trigger_history()`. This clears any event trigger history, ensuring that mobility event detection (such as handover triggers) starts fresh in the new cell context."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/set_current_cell",
    tags=[KnowledgeTag.UE, KnowledgeTag.MOBILITY, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/current_cell",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/serving_cell_history",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/execute_handover",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/cell_selection_and_camping",
        ),
    ],
)
def ue_set_current_cell_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "set_current_cell"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `set_current_cell` method updates the UE's association with a specific cell. "
        "It sets the `current_cell` attribute to the provided cell object and maintains a history of all serving cells in the `serving_cell_history` list.\n\n"
        "**Detailed behavior:**\n"
        "1. If the new cell is `None`, the method appends `None` to the `serving_cell_history`, indicating the UE is not currently served by any cell.\n"
        "2. If the new cell is not `None`, the method asserts that the UE is not already served by this cell (to avoid duplicate entries), then appends the cell's ID to the `serving_cell_history`.\n"
        "3. The `serving_cell_history` list is truncated to a maximum length (as defined by `UE_SERVING_CELL_HISTORY_LENGTH` in settings) to keep only the most recent associations.\n\n"
        "This method is called during cell selection, handover, and deregistration procedures to accurately track the UE's mobility and cell association history."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/deregister",
    tags=[
        KnowledgeTag.UE,
        KnowledgeTag.MOBILITY,
        KnowledgeTag.SIMULATION,
        KnowledgeTag.CODE,
    ],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_current_cell",
        ),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/connected",
        ),
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/user_equipments/methods/step"),
    ],
)
def ue_deregister_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "deregister"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `deregister` method removes the UE from the network and marks it as disconnected. "
        "This is typically called when the UE's simulation time expires or when it leaves the network.\n\n"
        "**Detailed behavior:**\n"
        "1. The method prints a deregistration message for logging.\n"
        "2. It notifies the current base station by calling `handle_deregistration_request(self)` on the serving base station, allowing the network to release resources associated with the UE.\n"
        "3. It calls `set_current_cell(None)` to clear the UE's current cell association and update the serving cell history.\n"
        "4. The `connected` attribute is set to `False`, indicating the UE is no longer part of the network.\n\n"
        "This method ensures a clean removal of the UE from the simulation, releasing all network resources and updating the UE's state."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/calculate_SINR_and_CQI",
    tags=[KnowledgeTag.UE, KnowledgeTag.QoS, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_sinr",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/set_downlink_cqi",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/downlink_received_power_dBm_dict",
        ),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/user_equipments/attributes/current_cell",
        ),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        ),
    ],
)
def ue_calculate_sinr_and_cqi_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "calculate_SINR_and_CQI"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `calculate_SINR_and_CQI` method computes the UE's downlink SINR (Signal-to-Interference-plus-Noise Ratio) and CQI (Channel Quality Indicator) "
        "based on the latest received power measurements from all detected cells.\n\n"
        "**Detailed behavior:**\n"
        "1. If the UE is not connected to any cell (`current_cell` is `None`), the method returns immediately.\n"
        "2. It retrieves the received power for the current cell from `downlink_received_power_dBm_dict`. If not available, it uses the cell's minimum required power.\n"
        "3. It calculates the total received power from all cells on the same frequency as the serving cell, converting dBm to Watts.\n"
        "4. The interference power is computed as the sum of all received powers on the same frequency, minus the serving cell's power.\n"
        "5. The thermal noise power is calculated using the Boltzmann constant, UE temperature, and serving cell bandwidth.\n"
        "6. The SINR is computed as the ratio of serving cell power to the sum of interference and noise, and converted to dB.\n"
        "7. The CQI is derived from the SINR using a mapping function (`sinr_to_cqi`).\n"
        "8. The method updates the UE's `downlink_sinr` and `downlink_cqi` attributes accordingly.\n\n"
        "This method is essential for link adaptation, resource allocation, and mobility decisions, as SINR and CQI directly impact data rates and handover triggers."
    )
    return code_block + explanation_text


@knowledge_entry(
    "/docs/user_equipments/methods/step",
    tags=[KnowledgeTag.UE, KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/move_towards_target",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/monitor_signal_strength",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/user_equipments/methods/check_rrc_meas_events_to_monitor",
        ),
        (KnowledgeRelationship.CALL_METHOD, "/docs/user_equipments/methods/deregister"),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/user_equipments/attributes/time_remaining",
        ),
    ],
)
def ue_step_explainer(sim, knowledge_router, query_key, params):
    try:
        method_source = inspect.getsource(getattr(UE, "step"))
        code_block = f"```python\n{method_source}\n```\n"
    except Exception:
        code_block = "*Source code unavailable.*\n\n"
    explanation_text = (
        "The `step` method advances the UE's state by one simulation time step. "
        "It orchestrates mobility, radio measurements, event monitoring, and lifecycle management for the UE.\n\n"
        "**Detailed behavior:**\n"
        "1. Calls `move_towards_target(delta_time)` to update the UE's position based on its speed and target coordinates.\n"
        "2. Calls `monitor_signal_strength()` to update the UE's received power measurements and recalculate SINR and CQI.\n"
        "3. Calls `check_rrc_meas_events_to_monitor()` to evaluate RRC measurement events and trigger handover or reporting if needed.\n"
        "4. Decrements the `time_remaining` attribute by the elapsed simulation time (`delta_time`).\n"
        "5. If `time_remaining` reaches zero or below, calls `deregister()` to remove the UE from the network and simulation.\n\n"
        "This method is typically called once per simulation tick, ensuring the UE's mobility, connectivity, and lifecycle are accurately modeled."
    )
    return code_block + explanation_text
