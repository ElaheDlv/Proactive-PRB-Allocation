from agents import Agent, ModelSettings  # RunConfig, Runner

# from intelligence_layer.user_chat.ue_chat_agent import ue_subscription_agent
# from .user_chat.model_recommender_agents import model_recommender_orchestrator
from .client_ai_service_pipeline_agents import client_ai_service_agent
import settings


# client_chat_agent = Agent(
#     name="Network Client Assistant",
#     instructions="""
#     You're the network client assistant. 
#     You chat with the network users/clients.
#     You only responsibility is to route user queries to appropriate agents you're connected with.
    
#     Routing Guidelines:
#     - Route all AI related questions to the AI Service agent
#     """,
#     handoffs=[client_ai_service_agent],
#     model=settings.OPENAI_NON_REASONING_MODEL_NAME,
#     model_settings=ModelSettings(
#         temperature=0,
#     ),
# )


# client_chat_agent = Agent(
#     name="User Assistant",
#     instructions="""
#     You chat directly with end users of the network. Users may request various tasks. There are two specific requests to handle as follows:

#         1. Subscription management (adding, updating, or removing subscriptions, e.g., for network services/SIMs/UEs)
#             - Always just hand off to the UE Subscription Management Agent, which processes user requests about starting new subscriptions, modifying which network slices a user/SIM can access, or removing subscriptions.
#             - Do not attempt to fulfill these requests yourself.

#         2. Provide a model recommendation
#             - Always just hand off to a suitable agent.
#             - Do not attempt to fulfill this request yourself.

#         Your Responsibilities:
#             - For the two tasks above: Immediately transfer the conversation to an agent.
#             - For all other general user queries: Handle and clarify these yourself. Do not hand off to an agent.

#     IMPORTANT: You should refer to "subscription management" tasks using that phrase, even for requests that mention "new UE", "new SIM", or "add a device" etc., since it is the correct network operations terminology.
#     """,
#     # handoffs=[model_recommender_orchestrator, ue_subscription_agent],
#     handoffs=[non_reasoning_network_knowledge_agent],
#     model=settings.OPENAI_NON_REASONING_MODEL_NAME,
#     model_settings=ModelSettings(
#         temperature=0,
#     ),
# )


# @cache
# def get_config():
#     model_settings = ModelSettings(
#         temperature=0,
#     )
#     return RunConfig(model_settings=model_settings, model="gpt-4.1")


# async def client_chat_agent_function(data):
#     """This function receives the user request sychronously, because it may require parsing in the ui to render"""
#     agent = client_chat_agent
#     if data.get("type") == "subscription":
#         agent = ue_subscription_agent

#     result = Runner.run_streamed(agent, data["chat"], run_config=get_config())
#     collected = []
#     async for event in result.stream_events():
#         if event.type == "raw_response_event" and isinstance(
#             event.data, ResponseTextDeltaEvent
#         ):
#             print(event.data.delta, end=" ", flush=True)
#             collected.append(event.data.delta)
#     result = "".join(collected)
#     try:
#         return json.loads(result)
#     except Exception:
#         return result
