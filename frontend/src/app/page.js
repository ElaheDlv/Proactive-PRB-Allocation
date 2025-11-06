"use client";

import { useState, useRef, useEffect } from "react";
import SimulationRenderer from "./components/SimulationRenderer";
import UEDashboard from "./components/UEDashboard";
import BaseStationDashboard from "./components/BaseStationDashboard";
import LogDashboard from "./components/LogDashboard";
import KnowledgeLayerDashboard from "./components/KnowledgeLayerDashboard";
import NetworkEngineerChat from "./components/NetworkEngineerChat";
import NetworkUserChat from "./components/NetworkUserChat/NetworkUserChat";

export default function Home() {
  const wsHost = process.env.NEXT_PUBLIC_WS_HOST || "localhost";
  const wsPort = process.env.NEXT_PUBLIC_WS_PORT || "8763";
  const wsProtocol = process.env.NEXT_PUBLIC_WS_PROTOCOL || "ws";
  const wsUrl = `${wsProtocol}://${wsHost}:${wsPort}`;
  const [websocket, setWebsocket] = useState(null);
  const [wsConnectionStatus, setWsConnectionStatus] = useState("disconnected");
  const [simulationState, setSimulationState] = useState(null);
  const [bottomTabListIndex, setBottomTabListIndex] = useState("ue_dashboard");
  const [rightTabListIndex, setRightTabListIndex] =
    useState("network_user_chat");
  const wsRef = useRef(null);
  const memoryRef = useRef([]);
  const messageHandlersRef = useRef({});

  const wsMessageHandler = (event) => {
    console.log("WebSocket message received:", event);
    if (event.data) {
      const messageData = JSON.parse(event.data);

      const { layer, command, response, error } = messageData;

      if (error) {
        console.error(`Error from ${layer}: ${error}`);
        return;
      }

      if (!layer || !command) {
        console.error("Invalid message format:", messageData);
        return;
      }

      // Check if a specific handler is registered for this layer and command
      const handlerKey = `${layer}_${command}`;
      const handlers = messageHandlersRef.current;
      if (handlers[handlerKey]) {
        console.log(`Handling message with registered handler: ${handlerKey}`);
        console.log(handlers[handlerKey]);
        handlers[handlerKey](response);
        return;
      } else {
        console.warn(
          `No handler registered for ${layer} command: ${command}. Nothing to do.`
        );
        console.log("Available handlers:", handlers);
      }
    }
  };

  const registerMessageHandler = (layer, command, handler) => {
    console.log("Registering message handler:", layer, command);
    messageHandlersRef.current[`${layer}_${command}`] = handler;
  };

  const deregisterMessageHandler = (layer, command) => {
    console.log("Deregistering message handler:", layer, command);
    delete messageHandlersRef.current[`${layer}_${command}`];
  };

  const connectWebSocket = () => {
    const ws = new WebSocket(wsUrl);
    setWebsocket(ws);

    ws.onopen = () => {
      setWsConnectionStatus("connected");
    };

    ws.onclose = () => {
      setWsConnectionStatus("disconnected");
    };

    ws.onerror = () => {
      setWsConnectionStatus("error");
    };

    ws.onmessage = wsMessageHandler;

    wsRef.current = ws;
  };

  const sendMessage = (layer, command, data = {}) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          layer,
          command,
          data,
        })
      );
    } else {
      console.error("WebSocket is not connected.");
    }
  };

  const onStartSimulation = () => {
    sendMessage("network_layer", "start_simulation");
  };

  const onStopSimulation = () => {
    sendMessage("network_layer", "stop_simulation");
  };

  const onGetSimulationState = () => {
    sendMessage("network_layer", "get_simulation_state");
  };

  useEffect(() => {
    registerMessageHandler(
      "network_layer",
      "simulation_state_update",
      (response) => {
        console.log("Simulation State Update:", response);
        setSimulationState(response);
        memoryRef.current.push(response);
        // Maintain fixed size of 1000
        if (memoryRef.current.length > 1000) {
          memoryRef.current.shift();
        }
      }
    );

    registerMessageHandler(
      "network_layer",
      "get_simulation_state",
      (response) => {
        console.log("Simulation State:", response);
      }
    );

    return () => {
      console.log("Cleaning up message handlers");
      deregisterMessageHandler("network_layer", "simulation_state_update");
      deregisterMessageHandler("network_layer", "get_simulation_state");
    };
  }, []);

  const saveMemoryToFile = () => {
    const dataStr = JSON.stringify(memoryRef.current, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "simulation_memory.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4 flex flex-col gap-2">
      <div className="flex flex-row gap-4 items-center">
        <button
          className="btn btn-outline"
          onClick={connectWebSocket}
          disabled={wsConnectionStatus === "connected"}
        >
          Connect To Simulation Server
        </button>
        <p>Status: {wsConnectionStatus}</p>

        <button
          className="btn btn-outline"
          onClick={onStartSimulation}
          disabled={wsConnectionStatus !== "connected"}
        >
          Start Simulation
        </button>

        <button
          className="btn btn-outline"
          onClick={onStopSimulation}
          disabled={wsConnectionStatus !== "connected"}
        >
          Stop Simulation
        </button>

        <button
          className="btn btn-outline"
          onClick={onGetSimulationState}
          disabled={wsConnectionStatus !== "connected"}
        >
          Get Simulation State
        </button>

        <button
          className="btn btn-outline"
          onClick={saveMemoryToFile}
          disabled={memoryRef.current.length === 0}
        >
          Save Memory to JSON
        </button>
      </div>
      <div className="flex flex-row gap-4">
        <SimulationRenderer simulationState={simulationState} />
        <div className="flex flex-col gap-3 h-[1000px] my-5 flex-1">
          <div role="tablist" className="tabs tabs-border">
            <a
              role="tab"
              className={
                "tab " +
                (rightTabListIndex === "network_engineer_chat"
                  ? "tab-active"
                  : "")
              }
              onClick={() => setRightTabListIndex("network_engineer_chat")}
            >
              Chat as Network Engineer
            </a>
            <a
              role="tab"
              className={
                "tab " +
                (rightTabListIndex === "network_user_chat" ? "tab-active" : "")
              }
              onClick={() => setRightTabListIndex("network_user_chat")}
            >
              Chat as Network User
            </a>
            <a
              role="tab"
              className={
                "tab " +
                (rightTabListIndex === "knowledge_dashboard"
                  ? "tab-active"
                  : "")
              }
              onClick={() => setRightTabListIndex("knowledge_dashboard")}
            >
              Knowledge Dashboard
            </a>
            <a
              role="tab"
              className={
                "tab " +
                (rightTabListIndex === "log_dashboard" ? "tab-active" : "")
              }
              onClick={() => setRightTabListIndex("log_dashboard")}
            >
              Log Dashboard
            </a>
          </div>

          <div className="flex-1 flex min-h-0">
            {rightTabListIndex === "network_user_chat" && (
              <NetworkUserChat
                sendMessage={sendMessage}
                registerMessageHandler={registerMessageHandler}
                deregisterMessageHandler={deregisterMessageHandler}
              />
            )}

            {rightTabListIndex === "network_engineer_chat" && (
              <NetworkEngineerChat
                registerMessageHandler={registerMessageHandler}
                sendMessage={sendMessage}
                deregisterMessageHandler={deregisterMessageHandler}
              />
            )}

            {rightTabListIndex === "knowledge_dashboard" && (
              <KnowledgeLayerDashboard
                sendMessage={sendMessage}
                registerMessageHandler={registerMessageHandler}
                deregisterMessageHandler={deregisterMessageHandler}
              />
            )}

            {rightTabListIndex === "log_dashboard" && (
              <LogDashboard simulationState={simulationState} />
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div role="tablist" className="tabs tabs-border">
          <a
            role="tab"
            className={
              "tab " +
              (bottomTabListIndex === "ue_dashboard" ? "tab-active" : "")
            }
            onClick={() => setBottomTabListIndex("ue_dashboard")}
          >
            UE Dashboard
          </a>
          <a
            role="tab"
            className={
              "tab " +
              (bottomTabListIndex === "base_station_dashboard"
                ? "tab-active"
                : "")
            }
            onClick={() => setBottomTabListIndex("base_station_dashboard")}
          >
            Base Station Dashboard
          </a>
        </div>
        <div className={bottomTabListIndex === "ue_dashboard" ? "" : "hidden"}>
          <UEDashboard simulationState={simulationState} />
        </div>
        <div
          className={
            bottomTabListIndex === "base_station_dashboard" ? "" : "hidden"
          }
        >
          <BaseStationDashboard simulationState={simulationState} />
        </div>
      </div>
    </div>
  );
}
