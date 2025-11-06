import json
import asyncio
from agents import ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall
from agents import Runner


class WebSocketSingleton:
    _instance = None
    _websocket = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketSingleton, cls).__new__(cls)
        return cls._instance

    def set_websocket(self, websocket):
        self._websocket = websocket

    def get_websocket(self):
        return self._websocket


class WebSocketResponse:
    def __init__(self, layer=None, command=None, response=None, error=None):
        self.layer = layer
        self.command = command
        self.response = response
        self.error = error

    def to_json(self):
        return json.dumps(
            {
                "layer": self.layer,
                "command": self.command,
                "response": self.response,
                "error": self.error,
            }
        )


async def handle_start_simulation(websocket, simulation_engine, knowledge_router, data):
    response = WebSocketResponse(
        layer="network_layer",
        command="start_simulation",
        response="Starting simulation",
        error=None,
    )
    await websocket.send(response.to_json())
    asyncio.create_task(simulation_engine.start_simulation())


async def handle_stop_simulation(websocket, simulation_engine, knowledge_router, data):
    response = WebSocketResponse(
        layer="network_layer",
        command="stop_simulation",
        response="Stopping simulation",
        error=None,
    )
    await websocket.send(response.to_json())
    simulation_engine.stop()


async def handle_get_simulation_state(
    websocket, simulation_engine, knowledge_router, data
):
    response = WebSocketResponse(
        layer="network_layer",
        command="get_simulation_state",
        response=simulation_engine.to_json(),
        error=None,
    )
    await websocket.send(response.to_json())


async def handle_get_routes(websocket, simulation_engine, knowledge_router, data):
    response = WebSocketResponse(
        layer="knowledge_layer",
        command="get_routes",
        response=knowledge_router.get_routes(),
        error=None,
    )
    await websocket.send(response.to_json())


async def handle_query_knowledge(websocket, simulation_engine, knowledge_router, data):
    response = WebSocketResponse(
        layer="knowledge_layer",
        command="query_knowledge",
        response=knowledge_router.query_knowledge(data),
        error=None,
    )
    await websocket.send(response.to_json())


# async def handle_network_user_chat(websocket, simulation_engine, knowledge_router, data):
#     result = await client_chat_agent_function(data)
#     response = WebSocketResponse(
#         layer="intelligence_layer_user",
#         command="chat_event_stream",
#         response={"event_type": "message_output_item", "message_output": result},
#         error=None,
#     )
#     await websocket.send(response.to_json())


async def stream_agent_chat(
    websocket, simulation_engine, knowledge_router, data, command, agent_func
):
    """
    Handles both 'network_user_chat' and 'network_engineer_chat' commands by streaming events
    and dispatching each event type to its own helper function.
    """
    chat_agent_streamer = Runner.run_streamed(agent_func, data)
    async for event in chat_agent_streamer.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await send_response_text_delta_event(websocket, command, event.data.delta)
        elif event.type == "agent_updated_stream_event":
            await send_agent_updated_event(websocket, command, event.new_agent.name)
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item" and isinstance(
                event.item.raw_item, ResponseFunctionToolCall
            ):
                await send_tool_call_item_event(
                    websocket,
                    command,
                    tool_name=event.item.raw_item.name,
                    tool_args=event.item.raw_item.arguments,
                )
            elif event.item.type == "tool_call_output_item":
                await send_tool_call_output_item_event(
                    websocket, command, tool_output=event.item.raw_item["output"]
                )
            elif event.item.type == "message_output_item":
                await send_message_output_item_event(
                    websocket,
                    command,
                    message_output=ItemHelpers.text_message_output(event.item),
                )
            # else: ignore other event types


# --- Helper functions for network_engineer_chat event types ---


async def send_response_text_delta_event(websocket, command, delta):
    response = WebSocketResponse(
        layer="intelligence_layer",
        command=command,
        response={
            "event_type": "response_text_delta_event",
            "response_text_delta": delta,
        },
        error=None,
    )
    await websocket.send(response.to_json())


async def send_agent_updated_event(websocket, command, agent_name):
    response = WebSocketResponse(
        layer="intelligence_layer",
        command=command,
        response={
            "event_type": "agent_updated_stream_event",
            "agent_name": agent_name,
        },
        error=None,
    )
    await websocket.send(response.to_json())


async def send_tool_call_item_event(websocket, command, tool_name, tool_args):
    response = WebSocketResponse(
        layer="intelligence_layer",
        command=command,
        response={
            "event_type": "tool_call_item",
            "tool_name": tool_name,
            "tool_args": tool_args,
        },
        error=None,
    )
    await websocket.send(response.to_json())


async def send_tool_call_output_item_event(websocket, command, tool_output):
    response = WebSocketResponse(
        layer="intelligence_layer",
        command=command,
        response={
            "event_type": "tool_call_output_item",
            "tool_output": tool_output,
        },
        error=None,
    )
    await websocket.send(response.to_json())


async def send_message_output_item_event(websocket, command, message_output):
    response = WebSocketResponse(
        layer="intelligence_layer",
        command=command,
        response={
            "event_type": "message_output_item",
            "message_output": message_output,
        },
        error=None,
    )
    await websocket.send(response.to_json())


async def handle_network_user_action(
    websocket, simulation_engine, knowledge_router, data
):
    action_type = data.get("action_type")
    response_content = None
    if action_type == "query_knowledge":
        query = data.get("query")
        response_content = {
            "action_type": action_type,
            "query_response": knowledge_router.query_knowledge(query),
        }
    else:
        response_content = {
            "action_type": action_type,
            "error": f"Unsupported action type: {action_type}",
        }
    response = WebSocketResponse(
        layer="intelligence_layer",
        command="network_user_action_response",
        response=response_content,
        error=None,
    )
    await websocket.send(response.to_json())
