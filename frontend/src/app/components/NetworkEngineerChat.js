"use client";

import React, { useState, useEffect, useRef } from "react";
import dayjs from "dayjs";
import Image from "next/image";
import agent_icon from "../../assets/agent_icon.png";
import tool_icon from "../../assets/tool_icon.png";

export default function NetworkEngineerChat({
  sendMessage,
  registerMessageHandler,
  deregisterMessageHandler,
}) {
  const [messages, setMessages] = useState([]);
  const [chatDisabled, setChatDisabled] = useState(false);
  const [input, setInput] = useState("");
  const messageContainerRef = useRef(null);

  useEffect(() => {
    if (registerMessageHandler) {
      registerMessageHandler(
        "intelligence_layer",
        "network_engineer_chat_response",
        (streamedChatEvent) => {
          if (!streamedChatEvent) return;

          const eventType = streamedChatEvent.event_type;

          if (eventType === "response_text_delta_event") {
            const response_text_delta = streamedChatEvent.response_text_delta;
            setMessages((prevMessages) => {
              const lastMessage = prevMessages[prevMessages.length - 1];
              if (lastMessage && lastMessage.role === "assistant") {
                return [
                  ...prevMessages.slice(0, -1),
                  {
                    ...lastMessage,
                    content: lastMessage.content + response_text_delta,
                    time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
                  },
                ];
              }
              return [
                ...prevMessages,
                {
                  role: "assistant",
                  content: response_text_delta,
                  time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
                },
              ];
            });
          } else if (eventType === "agent_updated_stream_event") {
            const agent_name = streamedChatEvent.agent_name;
            setMessages((prevMessages) => [
              ...prevMessages,
              {
                role: "monotone",
                content: agent_name,
                title: "Agent Activated:",
                time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
              },
            ]);
          } else if (eventType === "tool_call_item") {
            const tool_name = streamedChatEvent.tool_name;
            const tool_call_item = streamedChatEvent.tool_args;
            setMessages((prevMessages) => [
              ...prevMessages,
              {
                role: "monotone",
                content: `${tool_name}(${tool_call_item})`,
                title: "Tool Call",
                time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
              },
            ]);
          } else if (eventType === "tool_call_output_item") {
            const tool_output = streamedChatEvent.tool_output;
            setMessages((prevMessages) => [
              ...prevMessages,
              {
                role: "monotone",
                content: tool_output,
                title: "Tool Output",
                time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
              },
            ]);
          } else if (eventType === "message_output_item") {
            const message_output = streamedChatEvent.message_output;
            setMessages((prevMessages) => {
              const lastMessage = prevMessages[prevMessages.length - 1];
              if (!lastMessage || lastMessage.role !== "assistant") {
                return [
                  ...prevMessages,
                  {
                    role: "assistant",
                    content: message_output,
                    time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
                  },
                ];
              } else if (lastMessage.content !== message_output) {
                // replace the last message with the new one
                const updatedMessages = [...prevMessages];
                updatedMessages[updatedMessages.length - 1] = {
                  ...lastMessage,
                  content: message_output,
                  time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
                };
                return updatedMessages;
              }
              return prevMessages;
            });
            setChatDisabled(false);
          } else {
            console.log("Unknown event type:", eventType);
          }
        }
      );
    }

    return () => {
      if (deregisterMessageHandler) {
        deregisterMessageHandler(
          "intelligence_layer",
          "network_engineer_chat_response"
        );
      }
    };
  }, []);

  useEffect(() => {
    const container = messageContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    const chatHistory = [];
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      if (message.role === "user" || message.role === "assistant") {
        chatHistory.push({
          role: message.role,
          content: message.content,
        });
      } else if (message.role === "monotone") {
        chatHistory.push({
          role: "assistant",
          content: `${message.title} ${message.content}`,
        });
      }
    }
    chatHistory.push({
      role: "user",
      content: input,
    });
    sendMessage("intelligence_layer", "network_engineer_chat", chatHistory);
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: input,
        time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      },
    ]);
    setInput("");
    setChatDisabled(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderChatMessage = (msg, index) => {
    const isUser = msg.role === "user";
    const isAssistant = msg.role === "assistant";
    const isMonotone = msg.role === "monotone";
    const time = msg.time;

    return (
      <div key={index} className={`chat ${isUser ? "chat-end" : "chat-start"}`}>
        {(isAssistant ||
          (isMonotone && msg.title && msg.title === "Agent Activated:")) && (
          <div className="chat-image avatar">
            <Image
              alt="Agent Icon"
              src={agent_icon}
              className="w-10 h-10 object-cover"
              width={40}
              height={40}
            />
          </div>
        )}
        {isMonotone && msg.title && msg.title.includes("Tool") && (
          <div className="chat-image avatar">
            <Image
              alt="Tool Icon"
              src={tool_icon}
              className="w-10 h-10 object-cover"
              width={40}
              height={40}
            />
          </div>
        )}
        <div className="chat-header">
          {isUser ? "User" : isAssistant ? "Assistant" : msg.title}
          <time className="text-xs opacity-50">{time}</time>
        </div>
        <pre
          className={`chat-bubble whitespace-pre-wrap ${
            isUser
              ? "chat-bubble-info"
              : isAssistant
              ? "chat-bubble-success"
              : "bg-base-200 text-base-content"
          }`}
        >
          {msg.content}
        </pre>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full flex-1">
      {/* Chat Messages */}
      <div
        className="overflow-y-auto p-4 space-y-4 flex-1"
        ref={messageContainerRef}
      >
        {messages.map(renderChatMessage)}
      </div>

      {/* Input Box */}
      <div className="p-4 border-t border-base-300 bg-base-100 flex items-center gap-2">
        <button
          onClick={() => setMessages([])}
          className="btn btn-success h-full w-20"
          type="button"
        >
          <span className="text-xl">NEW</span>
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          className="textarea textarea-bordered flex-1 resize-none"
          rows={1}
        />
        <button
          onClick={handleSend}
          className="btn btn-primary h-full w-20 "
          disabled={chatDisabled}
        >
          <span className="text-xl">SEND</span>
        </button>
      </div>
    </div>
  );
}
