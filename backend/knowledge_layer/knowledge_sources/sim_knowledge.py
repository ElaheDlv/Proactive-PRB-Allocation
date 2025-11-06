import json
import inspect
from ..knowledge_entry import knowledge_entry
from ..tags import KnowledgeTag
from ..relationships import KnowledgeRelationship
from network_layer.simulation_engine import SimulationEngine

SUPPORTED_SIM_ATTRIBUTES = [
    "sim_started",
    "sim_step",
    "base_station_list",
    "cell_list",
    "ue_list",
    "global_UE_counter",
    "logs",
]

SUPPORTED_SIM_METHODS = [
    "network_setup",
    "spawn_random_ue",
    "add_ue",
    "spawn_UEs",
    "step_UEs",
    "remove_UE",
    "step_BSs",
    "step",
    "start_simulation",
    "stop",
    "to_json",
]


@knowledge_entry(
    key="/docs/sim_engine",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def sim_engine_docs(sim, knowledge_router, query_key, params):
    return (
        "📘 **Welcome to the Simulation Engine Knowledge Base**\n\n"
        "This knowledge base provides access to the core simulation engine that orchestrates the entire network simulation, "
        "including the management of base stations, cells, UEs, and simulation steps.\n\n"
        "You can query live data and explanations for the simulation engine.\n\n"
        "### Available Endpoints:\n"
        "- **List all attributes of the simulation engine**: `/sim_engine`\n"
        "- **Get a specific attribute value of the simulation engine**: `/sim_engine/attributes/{attribute_name}`\n"
        "- **Explain what an attribute of simulation engine means**: `/docs/sim_engine/attributess/{attribute_name}`\n"
        "- **Explain what a method in simulation engine class does**: `/docs/sim_engine/methods/{method_name}`\n"
        "### Supported SimulationEngine Attributes:\n"
        f"{', '.join(SUPPORTED_SIM_ATTRIBUTES)}\n\n"
        "### Supported SimulationEngine Methods:\n"
        f"{', '.join(SUPPORTED_SIM_METHODS)}\n\n"
    )


@knowledge_entry(
    key="/sim_engine",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def get_sim_engine_attributes(sim, knowledge_router, query_key, params):
    response = "Attributes value of the simulation engine:\n"
    for attr in SUPPORTED_SIM_ATTRIBUTES:
        value = getattr(sim, attr, None)
        if value is None:
            response += f"- {attr}: None\n"
            continue
        if attr == "ue_list" or attr == "base_station_list" or attr == "cell_list":
            response += f"{attr}:\n"
            for val in value.values():
                response += f"  - {repr(val)}\n"
        else:
            response += f"- {attr}: {repr(value)}\n"

    return response


# ------------------------------------------
#     GET /user_equipments/{ue_imsi}/attributes/{attribute_name}
#       → Get a specific attribute value for the given UE
# ------------------------------------------
@knowledge_entry(
    key="/sim_engine/attributes/{attribute_name}",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def get_sim_engine_attribute_value(sim, knowledge_router, query_key, params):
    attribute_name = params["attribute_name"]
    if attribute_name not in SUPPORTED_SIM_ATTRIBUTES:
        return f"Attribute '{attribute_name}' is not supported by the simulation engine class."
    value = getattr(sim, attribute_name, None)
    response = f"Value of attribute '{attribute_name}':"
    if value is None:
        response += " None"
    elif (
        attribute_name == "ue_list"
        or attribute_name == "base_station_list"
        or attribute_name == "cell_list"
    ):
        response += "\n"
        for val in value.values():
            response += f"  - {repr(val)}\n"
    else:
        response += f" {repr(value)}"
    return response


@knowledge_entry(
    "/docs/sim_engine/attributes/sim_started",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def sim_started_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `sim_started` attribute is a boolean flag indicating whether the simulation is currently running. "
        "It is set to `True` when the simulation is started and `False` when stopped or not yet started. "
        "This flag is used to control the main simulation loop and prevent multiple concurrent runs."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/sim_step",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def sim_step_attribute_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `sim_step` attribute tracks the current time step of the simulation. "
        "It is incremented at each iteration of the simulation loop."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/base_station_list",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.BS],
    related=[
        (KnowledgeRelationship.HAS_ATTRIBUTE, "/docs/base_station"),
    ],
)
def sim_base_station_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `base_station_list` attribute is a dictionary mapping base station IDs to BaseStation objects managed by the simulation engine. "
        "It provides centralized access to all base stations in the simulation and is used for network setup, resource allocation, and mobility management."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/cell_list",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.CELL],
    related=[
        (KnowledgeRelationship.HAS_ATTRIBUTE, "/docs/cell"),
    ],
)
def sim_cell_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `cell_list` attribute is a dictionary mapping cell IDs to Cell objects managed by the simulation engine. "
        "It allows the simulation to efficiently access and update the state of all cells in the network."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/ue_list",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE],
    related=[
        (KnowledgeRelationship.HAS_ATTRIBUTE, "/docs/user_equipments"),
    ],
)
def sim_ue_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ue_list` attribute is a dictionary mapping UE IMSIs to UE objects currently present in the simulation. "
        "It is updated as UEs are spawned, move, or deregister, and is essential for tracking all active UEs in the network."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/global_UE_counter",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE],
    related=[],
)
def sim_global_UE_counter_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `global_UE_counter` attribute is an integer counter that tracks the total number of UEs ever created in the simulation. "
        "It is used to assign unique IMSIs to new UEs and ensures that each UE can be uniquely identified."
    )


@knowledge_entry(
    "/docs/sim_engine/attributes/logs",
    tags=[KnowledgeTag.SIMULATION],
    related=[],
)
def sim_logs_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `logs` attribute is a list of log messages generated during the simulation. "
        "It records significant events, errors, and state changes, providing a trace of simulation activity for debugging and analysis."
    )


# Method explainers


@knowledge_entry(
    "/docs/sim_engine/methods/network_setup",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/sim_engine/attributes/core_network",
        ),
        (
            KnowledgeRelationship.CALL_METHOD,
            "/docs/sim_engine/methods/add_base_station",
        ),
        (KnowledgeRelationship.SET_ATTRIBUTE, "/docs/sim_engine/attributes/ric"),
    ],
)
def sim_network_setup_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "network_setup"))
    explanation = (
        "The `network_setup` method initializes the core network, base stations, and the RAN Intelligent Controller (RIC) for the simulation. "
        "It reads configuration data, creates all required network elements, and ensures that the simulation environment is ready for operation. "
        "This method must be called before starting the simulation to ensure all components are properly instantiated and connected."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/spawn_random_ue",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALL_METHOD, "/docs/user_equipments/methods/power_up"),
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/add_ue"),
    ],
)
def sim_spawn_random_ue_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "spawn_random_ue"))
    explanation = (
        "The `spawn_random_ue` method creates a new UE at a random position within the simulation boundaries, assigns it a random target and speed, "
        "and attempts to power it up and register it with the network. If successful, the UE is added to the simulation's UE list. "
        "This method is used to dynamically introduce new UEs into the simulation, supporting scenarios with varying UE density and mobility."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/add_ue",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/sim_engine/methods/spawn_random_ue",
        ),
        (KnowledgeRelationship.SET_ATTRIBUTE, "/docs/sim_engine/attributes/ue_list"),
    ],
)
def sim_add_ue_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "add_ue"))
    explanation = (
        "The `add_ue` method adds a UE object to the simulation's UE list and increments the global UE counter. "
        "It ensures that each UE is uniquely identified and tracked throughout its lifecycle in the simulation."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/spawn_UEs",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/spawn_random_ue"),
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/sim_engine/methods/step"),
    ],
)
def sim_spawn_UEs_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "spawn_UEs"))
    explanation = (
        "The `spawn_UEs` method determines how many new UEs to introduce in the current simulation step, based on configuration and randomization. "
        "It repeatedly calls `spawn_random_ue` to create and register new UEs, up to the maximum allowed UE count. "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/step_UEs",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/user_equipments/methods/step"),
        (KnowledgeRelationship.USES_ATTRIBUTE, "/docs/sim_engine/attributes/ue_list"),
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/remove_UE"),
    ],
)
def sim_step_UEs_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "step_UEs"))
    explanation = (
        "The `step_UEs` method advances the state of all UEs in the simulation by one time step. "
        "It calls each UE's `step` method to update its position, connectivity, and state, and removes UEs that are no longer connected. "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/remove_UE",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.UE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/sim_engine/methods/step_UEs"),
        (KnowledgeRelationship.SET_ATTRIBUTE, "/docs/sim_engine/attributes/ue_list"),
    ],
)
def sim_remove_UE_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "remove_UE"))
    explanation = (
        "The `remove_UE` method removes a specified UE from the simulation's UE list and logs the deregistration event. "
        "It is used to cleanly remove UEs that have completed their session or are no longer active in the simulation."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/step_BSs",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.BS, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALLED_BY_METHOD, "/docs/sim_engine/methods/step"),
        (
            KnowledgeRelationship.USES_ATTRIBUTE,
            "/docs/sim_engine/attributes/base_station_list",
        ),
        (KnowledgeRelationship.CALL_METHOD, "/docs/base_station/methods/step"),
    ],
)
def sim_step_BSs_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "step_BSs"))
    explanation = (
        "The `step_BSs` method advances the state of all base stations in the simulation by one time step. "
        "It calls each base station's `step` method, which in turn updates all managed cells and processes control actions. "
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/step",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/sim_engine/methods/start_simulation",
        ),
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/spawn_UEs"),
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/step_UEs"),
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/step_BSs"),
    ],
)
def sim_step_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "step"))
    explanation = (
        "The `step` method orchestrates a single simulation time step. "
        "It performs the following actions in sequence: spawns new UEs if needed, updates all UEs, and advances all base stations. "
        "This method is called repeatedly during the simulation loop and is responsible for driving the overall progression of the simulation."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/start_simulation",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.CALL_METHOD, "/docs/sim_engine/methods/step"),
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/sim_engine/attributes/sim_started",
        ),
    ],
)
def sim_start_simulation_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "start_simulation"))
    explanation = (
        "The `start_simulation` method is an asynchronous coroutine that starts the main simulation loop. "
        "It repeatedly calls the `step` method, sends simulation state updates, and advances the simulation time until the maximum number of steps is reached or the simulation is stopped. "
        "This method is the entry point for running the simulation and coordinating all network activities."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


@knowledge_entry(
    "/docs/sim_engine/methods/stop",
    tags=[KnowledgeTag.SIMULATION, KnowledgeTag.CODE],
    related=[
        (
            KnowledgeRelationship.SET_ATTRIBUTE,
            "/docs/sim_engine/attributes/sim_started",
        ),
    ],
)
def sim_stop_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(SimulationEngine, "stop"))
    explanation = "The `stop` method pause the simulation by setting the `sim_started` flag to `False` and logging the stop event. "
    return f"```python\n{code}\n```\n\n{explanation}"
