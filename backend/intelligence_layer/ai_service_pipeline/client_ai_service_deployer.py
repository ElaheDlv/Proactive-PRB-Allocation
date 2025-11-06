import asyncio
from utils import WebSocketSingleton, WebSocketResponse

from agents import Agent, function_tool
from network_layer.simulation_engine import SimulationEngine
from knowledge_layer import KnowledgeRouter


@function_tool
def deploy_ai_service(ai_service_name: str, ue_ids: list[str]) -> str:
    """Deploys an AI service for specified User Equipment (UE) IDs.

    Args:
        ai_service_name (str): The name of the AI service to deploy.
        ue_ids (list[str]): A list of User Equipment IDs in the format `IMSI_<digits>`.
    """
    websocket = WebSocketSingleton().get_websocket()
    simulation_engine = SimulationEngine()
    knowledge_router = KnowledgeRouter()

    ai_service_data = knowledge_router.query_knowledge(
        f"/ai_services/{ai_service_name}/raw",
    )

    if not ai_service_data:
        print(f"AI service {ai_service_name} is not supported.")
        return f"AI service {ai_service_name} is not supported. Please check the supported services by querying /ai_services at the knowledge database."

    if websocket is None:
        print("WebSocket is not available.")
        return "Websocket connection with the frontend is not available."

    if not simulation_engine.ric:
        print("RIC is not initialized in the simulation engine.")
        return "RIC is not initialized in the simulation engine. Unable to deploy AI service."

    new_ai_service_subscription = (
        simulation_engine.ric.ai_service_subscription_manager.create_subscription(
            ai_service_name=ai_service_name,
            ai_service_data=ai_service_data,
            ue_id_list=ue_ids,
        )
    )

    if new_ai_service_subscription is None:
        return "Failed to deploy AI service due to an internal error."

    response_message = f"""AI service subscription created successfully with ID: {new_ai_service_subscription.subscription_id} for service {ai_service_name} with UEs {ue_ids}.

When the User Equipments are connected to any base station,
the AI service will be started automatically at the edge clustered of that base station if computation resources are available.

The AI services offers RESTful API endspoint at `http://cranfield_6G.com/ai_services/{ai_service_name}`. 
The Base Stations will automatically perform local breakout of UE's requests and route them to the AI services at the edge clusters.

Below are the sample code snippets for using the AI service:

```python
{ai_service_data["code"]["ai_client_script_content"]}
```
"""
    response = WebSocketResponse(
        layer="intelligence_layer",
        command="ai_service_pipeline_response",
        response={
            "event_type": "ai_service_deployment",
            "subscription_data": new_ai_service_subscription.to_json(),
            "message": response_message,
        },
        error=None,
    )
    asyncio.create_task(websocket.send(response.to_json()))

    return response_message


client_ai_service_deployer = Agent(
    name="Client AI Service Deployer Assistant",
    handoff_description="Specialist agent for deploying AI services across our network's edge clusters.",
    tools=[
        deploy_ai_service,
    ],
    instructions=f"""You're the second step in helping users to select and deploy the right AI services for their devices or applications connected to our simulated network.

The user has just selected the AI service for deployment.
You follow steps below to complete your task:

1. Ask the user for his/her User Equipment IDs. The UE IDs are in the format of `IMSI_<digits>`, where `<digits>` is a sequence of digits. The user can provide multiple UE IDs.
2. Call the `deploy_ai_service` function tool with the selected AI service name and the provided UE IDs.

The tool will return instructions for users to use the AI service from their devices or applications, 
you need to forward the instructions back to the user.""",
)
