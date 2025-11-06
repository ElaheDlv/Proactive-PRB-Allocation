"use client";

import React, { useState, useEffect, useRef } from "react";
import UserMenuMain from "./UserMenuMain";
import UserMenuAIServiceRequest from "./UserMenuAiServiceRequest";
import dayjs from "dayjs";
import renderMessage from "./MessageRenderer";

function getDefaultMessages() {
  return [
    {
      role: "monotone",
      title: "Monotone",
      content: `Thank you for choosing our network.
  
We offer a range of AI services ready-to-deploy across our base stations for your connected user equipments.
Simply describe what applications or use cases you have in mind or you're currently building,
and our AI assistant will help you deploy the services you need across our edge/cloud clusters.`,
      time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
    },
  ];
}

export default function NetworkUserChat({
  sendMessage,
  registerMessageHandler,
  deregisterMessageHandler,
}) {
  const [currentMenu, setCurrentMenu] = useState("main_menu");
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null); // Add this ref

  let renderedMenu = null;

  switch (currentMenu) {
    case "main_menu":
      renderedMenu = (
        <UserMenuMain
          registerMessageHandler={registerMessageHandler}
          deregisterMessageHandler={deregisterMessageHandler}
          getDefaultMessages={getDefaultMessages}
          sendMessage={sendMessage}
          setMessages={setMessages}
          setCurrentMenu={setCurrentMenu}
        />
      );
      break;
    case "request_ai_service_menu":
      renderedMenu = (
        <UserMenuAIServiceRequest
          registerMessageHandler={registerMessageHandler}
          deregisterMessageHandler={deregisterMessageHandler}
          sendMessage={sendMessage}
          setMessages={setMessages}
          setCurrentMenu={setCurrentMenu}
        />
      );
      break;
    default:
      renderedMenu = null;
      break;
  }

  useEffect(() => {
    console.log("Messages updated:", messages);
    // Scroll to bottom when messages update
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollTop = messagesEndRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full flex-1">
      {/* Messages */}
      <div
        className="overflow-y-auto p-4 space-y-4 flex-1 text-2xl"
        ref={messagesEndRef}
      >
        {messages.map((msg, index) => renderMessage(index, msg))}
      </div>
      {renderedMenu}
    </div>
  );
}
