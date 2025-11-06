import Image from "next/image";
import agent_icon from "../../../assets/agent_icon.png";
import tool_icon from "../../../assets/tool_icon.png";
import thinking_icon from "../../../assets/thinking_icon.png";

export default function renderMessage(index, msg) {
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
      {isMonotone && msg.title && msg.title.includes("Monotone") && (
        <div className="chat-image avatar">
          <Image
            alt="Thinking Icon"
            src={thinking_icon}
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
        
      <div
        className={`chat-bubble whitespace-pre-wrap ${
          isUser
            ? "chat-bubble-info"
            : isAssistant
            ? "chat-bubble-success"
            : "bg-base-200 text-base-content"
        }`}
      >
        {msg.content}
      </div>
    </div>
  );
}
