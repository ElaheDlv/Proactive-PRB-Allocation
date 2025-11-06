import asyncio
from utils import WebSocketSingleton, WebSocketResponse

from agents import Agent, function_tool
from ..knowledge_tools import get_knowledge, get_knowledge_bulk
from settings import OPENAI_NON_REASONING_MODEL_NAME
from knowledge_layer import KnowledgeRouter


@function_tool
def recommend_ai_services(ai_service_names: list[str]) -> str:
    """Recommend AI services for the user to select from.

    Args:
        ai_service_names (list[str]): A list of AI service names to recommend.
    """

    knowledge_router = KnowledgeRouter()
    websocket = WebSocketSingleton().get_websocket()

    if websocket is None:
        print("WebSocket is not available.")
        return "Websocket connection with the frontend is not available."

    ai_service_descriptions = [
        knowledge_router.query_knowledge(f"/ai_services/{name}")
        for name in ai_service_names
    ]

    response = WebSocketResponse(
        layer="intelligence_layer",
        command="ai_service_pipeline_response",
        response={
            "event_type": "ai_services_recommendation",
            "ai_service_names": ai_service_names,
            "ai_service_descriptions": ai_service_descriptions,
        },
        error=None,
    )
    asyncio.create_task(websocket.send(response.to_json()))

    return "AI services recommendation sent to the user."


client_ai_service_need_profiler = Agent(
    name="Client AI Service Need Profiler Assistant",
    handoff_description="Specialist agent for understanding user's real AI service needs and intentions.",
    tools=[
        get_knowledge,
        get_knowledge_bulk,
        recommend_ai_services,
    ],
    # instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    instructions=f"""You're the the first step in helping users to select and deploy the right AI services for their devices and applications connected to our simulated network.

You're equipped with an AI service knowledge base, containg ready-to-deploy AI services across our simulated network.
The "get knowledge (bulk)" tools allow you to query the AI service database containing ready-to-deploy AI services across our simulated network.
To increase efficiency, you can use the bulk query tools to query multiple keys at once.

Always use the knowledge from the AI services knowledge base to answer user requests or questions.
You can start with the query key "/docs/ai_services" for the guidance to better explore the AI services knowledge base.  

Your job is to chat with the user and guide the user to find a suitable AI service in our knowledge base for their needs.
Please focus only on the use case's AI service needs instead of other non-AI functionalities such as logging, monitorings, etc.

Use the tool `recommend_ai_services` to recommend AI services based on the user's needs and preferences to allow the user to select from a list of AI services.
""",
    model=OPENAI_NON_REASONING_MODEL_NAME,
)
