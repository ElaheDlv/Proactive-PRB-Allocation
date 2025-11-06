import json
import inspect
from ..tags import KnowledgeTag
from ..relationships import KnowledgeRelationship
from network_layer.ric import RIC
from ..knowledge_entry import knowledge_entry

from network_layer.xApps.xapp_base import xAppBase


SUPPORTED_RIC_ATTRIBUTES = [
    "ric_id",
    "xapp_list",
]

SUPPORTED_RIC_METHODS = [
    "load_xApps",
]


@knowledge_entry(
    key="/docs/ric",
    tags=[KnowledgeTag.RIC, KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def ric_knowledge_help(sim, knowledge_router, query_key, params):
    return (
        "📘 **Welcome to the RAN Intelligent Controller (RIC) Knowledge Base**\n\n"
        "You can query live data and explanations for RIC in the simulation.\n\n"
        "### Available Endpoints:\n"
        "- **Get all attributes values for the simulated RIC**: `/ric`\n"
        "- **Get a specific attribute value of the RIC**: `/ric/attributes/{attribute_name}`\n"
        "- **Explain what an attribute of the RIC means**: `/docs/ric/attributes/{attribute_name}`\n"
        "- **Explain what a method in RIC class does**: `/docs/user_equipments/methods/{method_name}`\n"
        "- **List all the activated xApps (identifiers only) in the RIC**: `/docs/ric/xapps`\n"
        "- **Explain what a specific xApp does**: `/docs/ric/xapps/{xapp_name}`\n\n"
        "### Supported RIC Attributes:\n"
        f"{', '.join(SUPPORTED_RIC_ATTRIBUTES)}\n\n"
        "### Supported RIC Methods:\n"
        f"{', '.join(SUPPORTED_RIC_METHODS)}\n\n"
    )


# ------------------------------------------
#     GET /ric
#       → List all attributes values for the RIC
# ------------------------------------------
@knowledge_entry(
    key="/ric",
    tags=[KnowledgeTag.RIC],
    related=[],
)
def get_ric_attributes(sim, knowledge_router, query_key, params):
    ric = getattr(sim, "ric", None)
    if not ric:
        return "RAN Intelligent Controller has not been initiated yet."
    response = f"Attributes of RIC:\n"
    for attr in SUPPORTED_RIC_ATTRIBUTES:
        value = getattr(ric, attr, None)
        if value is None:
            response += f"- {attr}: None\n"
            continue
        if attr == "xapp_list":
            response += f"- {attr}: \n"
            for xapp_key in value.keys():
                response += f"  - {xapp_key}\n"
            continue

        response += f"- {attr}: {repr(value)}\n"

    return response


@knowledge_entry(
    key="/ric/attributes/{attribute_name}",
)
def ric_attribute_getter(sim, knowledge_router, query_key, params):
    ric = getattr(sim, "ric", None)
    if not ric:
        return "RAN Intelligent Controller has not been initiated yet."
    attribute_name = params["attribute_name"]
    if attribute_name not in SUPPORTED_RIC_ATTRIBUTES:
        return f"Attribute '{attribute_name}' is not supported. Supported attributes: {', '.join(SUPPORTED_RIC_ATTRIBUTES)}"
    value = getattr(ric, attribute_name, None)
    if value is None:
        return f"Attribute '{attribute_name}' is not set or does not exist."

    if attribute_name == "xapp_list":
        xapp_ids = list(value.keys())
        return f"Value of '{attribute_name}':\n" + "\n".join(
            f"- {xapp_id}" for xapp_id in xapp_ids
        )

    return f"Value of '{attribute_name}': {repr(value)}"


@knowledge_entry(
    "/docs/ric/attributes/ric_id",
    tags=[KnowledgeTag.KNOWLEDGE_GUIDE],
    related=[],
)
def ric_id_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `ric_id` attribute is a unique identifier for the Near Real-Time RAN Intelligent Controller (NearRT RIC) instance. "
        "It distinguishes this RIC from other possible RICs in a multi-controller simulation environment."
    )


@knowledge_entry(
    "/docs/ric/attributes/xapp_list",
    tags=[KnowledgeTag.KNOWLEDGE_GUIDE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.SET_BY_METHOD, "/docs/ric/methods/load_xApps"),
    ],
)
def ric_xapp_list_explainer(sim, knowledge_router, query_key, params):
    return (
        "The `xapp_list` attribute is a dictionary mapping xApp IDs to their instantiated xApp objects. "
        "Each xApp implements specific RAN control logic (such as handover, load balancing, etc.) and is dynamically loaded by the RIC at runtime. "
        "This attribute allows the RIC to manage, start, and coordinate all xApps in the simulation."
    )


# Method explainers


@knowledge_entry(
    "/docs/ric/methodss/load_xApps",
    tags=[KnowledgeTag.KNOWLEDGE_GUIDE, KnowledgeTag.CODE],
    related=[
        (KnowledgeRelationship.SET_ATTRIBUTE, "/docs/ric/attributes/xapp_list"),
        (
            KnowledgeRelationship.CALLED_BY_METHOD,
            "/docs/sim_engine/methods/network_setup",
        ),
    ],
)
def ric_load_xapps_explainer(sim, knowledge_router, query_key, params):
    code = inspect.getsource(getattr(RIC, "load_xApps"))
    explanation = (
        "The `load_xApps` method dynamically discovers and loads all available xApps from the xApps directory. "
        "It instantiates each xApp, assigns it to the RIC, and stores it in the `xapp_list` attribute. "
        "After loading, it calls the `start` method on each xApp to initialize their operation. "
        "This enables modular, plug-and-play RAN control logic, allowing new xApps to be added to the simulation without modifying the RIC core code. "
        "The method also ensures that no duplicate xApp IDs are loaded, maintaining the integrity of the xApp registry."
    )
    return f"```python\n{code}\n```\n\n{explanation}"


# --- xApp knowledge routes ---


@knowledge_entry(
    key="/docs/ric/xapps",
    tags=[KnowledgeTag.KNOWLEDGE_GUIDE, KnowledgeTag.RIC],
    related=[
        (KnowledgeRelationship.SET_BY_METHOD, "/docs/ric/methods/load_xApps"),
    ],
)
def ric_xapps_root(sim, knowledge_router, query_key, params):
    ric = getattr(sim, "ric", None)
    if not ric or not hasattr(ric, "xapp_list"):
        return "RIC or xApp list not found."
    xapp_ids = list(ric.xapp_list.keys())
    try:
        base_code = inspect.getsource(xAppBase)
    except Exception:
        base_code = "Source code unavailable."
    base_doc = (
        inspect.getdoc(xAppBase) or "No docstring/documentation available for xAppBase."
    )
    return (
        "xApps are modular applications loaded by the NearRT RIC to implement specific RAN control logic (e.g., handover, load balancing, etc.).\n"
        "The following xApps are currently activated:\n"
        f"{json.dumps(xapp_ids, indent=2)}\n\n"
        "Query `/ric/xapps/{xapp_id}` to view the source code and explanation for a specific xApp.\n\n"
        "## xAppBase (Base class for all xApps)\n"
        f"**Documentation:**\n{base_doc}\n\n"
        f"**Source code:**\n```python\n{base_code}\n```"
    )


def ric_xapp_detail(sim, knowledge_router, query_key, params):
    ric = getattr(sim, "ric", None)
    if not ric or not hasattr(ric, "xapp_list"):
        return "RIC or xApp list not found."
    xapp_id = params["xapp_id"]
    xapp = ric.xapp_list.get(xapp_id)
    if not xapp:
        return f"xApp '{xapp_id}' not found. Loaded xApps: {list(ric.xapp_list.keys())}"
    # Get source code
    try:
        code = inspect.getsource(xapp.__class__)
    except Exception:
        code = "Source code unavailable."
    doc = (
        inspect.getdoc(xapp.__class__)
        or "No docstring/documentation available for this xApp."
    )
    return (
        f"## xApp: {xapp_id}\n\n"
        f"**Documentation:**\n{doc}\n\n"
        f"**Source code:**\n```python\n{code}\n```"
    )


knowledge_entry(
    key="/docs/ric/xapps/{xapp_id}",
    tags=[KnowledgeTag.RIC],
    related=[],
)(ric_xapp_detail)

knowledge_entry(
    key="/ric/xapps/{xapp_id}",
    tags=[KnowledgeTag.RIC],
    related=[],
)(ric_xapp_detail)
