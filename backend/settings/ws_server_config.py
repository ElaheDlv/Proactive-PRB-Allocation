# ---------------------------
# Websocket Server Configuration
# ---------------------------
import os

WS_SERVER_HOST = os.getenv("WS_SERVER_HOST", "localhost")
try:
    WS_SERVER_PORT = int(os.getenv("WS_SERVER_PORT", "8763"))
except Exception:
    WS_SERVER_PORT = 8763
