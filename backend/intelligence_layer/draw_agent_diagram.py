# to execute: cd backend; python -m intelligence_layer.draw_agent_diagram

from agents.extensions.visualization import draw_graph

# from backend.intelligence_layer.OLD_client_chat_agent import client_chat_agent
from intelligence_layer.engineer_chat_agent import engineer_chat_agent


draw_graph(engineer_chat_agent, filename="engineer_agent_graph.dot")
# draw_graph(client_chat_agent, filename="user_agent_graph.dot")
