from agents import Agent
from .knowledge_tools import get_knowledge, get_knowledge_bulk
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from settings import OPENAI_NON_REASONING_MODEL_NAME, OPENAI_REASONING_MODEL_NAME


print(f"Using reasoning model: {OPENAI_REASONING_MODEL_NAME}")
print(f"Using non-reasoning model: {OPENAI_NON_REASONING_MODEL_NAME}")


non_reasoning_network_knowledge_agent = Agent(
    name="Basic Network Knowledge Assistant",
    handoff_description="Specialist agent for querying the telecom network knowledge database using tools",
    tools=[
        get_knowledge,
        get_knowledge_bulk,
    ],
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Use the tools to query the knowledge database of the simulated telecom network. 
To increase efficiency, you can use the bulk query tools to query multiple keys at once.
You should always start with 
    "/docs/user_equipments" (documentation on UE-related knowledge base) 
    "/docs/base_stations" (documentation on Base Station-related knowledge base)
    "/docs/cells" (documentation on Cell-related knowledge base)
    "/docs/ric" (documentation on RIC-related knowledge base)
    "/docs/sim_engine" (documentation on simulation execution-related knowledge base)
    "/docs/ai_services" (documentation on AI service-related knowledge base)

The knowledge tools often returns a list of related knowledge keys.
You should explore these related knowledge keys as well to gather more information to answer the user query wherever possible.
Note that most elements in the query keys are plural (user_equipments, base_stations, cells, ai_services, attributes, methods),
except for the RIC and simulation engine, which are singular (ric, sim_engine).
""",
    model=OPENAI_NON_REASONING_MODEL_NAME,
)


reasoning_network_knowledge_agent = Agent(
    name="Basic Network Knowledge Assistant",
    handoff_description="Specialist agent for querying the telecom network knowledge database using tools",
    tools=[
        get_knowledge,
        get_knowledge_bulk,
    ],
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
Use the tools to query the knowledge database of the simulated telecom network. 
To increase efficiency, you can use the bulk query tools to query multiple keys at once.
You should always start with 
    "/docs/user_equipments" (documentation on UE-related knowledge base) 
    "/docs/base_stations" (documentation on base station-related knowledge base)
    "/docs/cells" (documentation on cell-related knowledge base)
    "/docs/ric" (documentation on RIC-related knowledge base)
    "/docs/sim_engine" (documentation on simulation-related knowledge base)

The knowledge tools often returns a list of related knowledge keys.
You should explore these related knowledge keys as well to gather more information to answer the user query wherever possible.
Note that most elements in the URL patterns are plural (user_equipments, base_stations, cells, attributes, methods),
except for the RIC and simulation engine, which are singular (ric, sim_engine).
""",
    model=OPENAI_NON_REASONING_MODEL_NAME,
)
