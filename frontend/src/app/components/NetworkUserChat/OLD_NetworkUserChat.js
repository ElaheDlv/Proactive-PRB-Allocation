'use client';

import React, { useState, useEffect, useRef } from "react";
import dayjs from "dayjs";
import Image from "next/image";
// import { UESelector } from "../../UESelector";
// import ThinkingMessage from "../../ThinkingMessage";
// import CuratedConfigMessage from "../../CuratedConfigMessage";
import agent_icon from "../../../assets/agent_icon.png";
import tool_icon from "../../../assets/tool_icon.png";

const getDefaultWelcomeMessage = () => ({
  role: "assistant",
  content: `Thank you for choosing our network.

We offer a range of AI services ready-to-deploy across our base stations for your connected user equipments.
Simply describe what applications or use cases you have in mind or you're currently building,
and our AI assistant will help you deploy the services you need across our edge/cloud clusters.`,
  time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
});

/**
 * NetworkUserChat component: Provides the main chat interface for network users to interact
 * with the AI assistant, including message display, input, and handling
 * specific message types like UE selection and curated configurations.
 * @param {function} sendMessage - Function to send messages to the backend.
 * @param {object} streamedChatEvent - Object containing the latest streamed chat event.
 */
export default function NetworkUserChat({ sendMessage, streamedChatEvent }) {
  // State for managing chat messages
  const [messages, setMessages] = useState([getDefaultWelcomeMessage()]);

  // State to disable chat input and buttons during processing (also used to lock full session after deployment)
  const [chatDisabled, setChatDisabled] = useState(false);

  // State to store the IMSIs of UEs selected via the UESelector
  // const [selectedUEIMSI, setSelectedUEIMSI] = useState([]);

  // State for the current input text in the chat box
  const [input, setInput] = useState("");
  // Ref for the message container to enable auto-scrolling
  const messageContainerRef = useRef(null);

  // // State to store the last curated configuration (for deployment with subscriptions)
  // const [lastCuratedConfig, setLastCuratedConfig] = useState(null);

  // // State to backup the latest curated config deployment selection
  // const [curatedSelection, setCuratedSelection] = useState(null);

  // State to control visibility of the "Clear Chat" warning modal
  const [showWarning, setShowWarning] = useState(false);

  /**
   * Clears the chat messages and resets the chat state.
   */
  const clearChat = () => {
    setMessages([getDefaultWelcomeMessage()]); // Clear all messages
    setChatDisabled(false); // Enable chat
    setShowWarning(false); // Hide warning modal
    // setSelectedUEIMSI([]); // Clear selected UEs
  };

  /**
   * Confirms the clear chat action after the warning is accepted.
   * Just clears the chat.
   */
  const confirmClearChat = () => {
    setMessages([getDefaultWelcomeMessage()]); // Reset to welcome message only
    setChatDisabled(false); // Enable chat
    setShowWarning(false); // Hide warning
    setSelectedUEIMSI([]); // Clear selected UEs
  };

  /**
   * Cancels the clear chat action and hides the warning modal.
   */
  const cancelClearChat = () => {
    setShowWarning(false); // Hide warning
    setTempSelectedButton(null); // Clear temporary button
  };

  /**
   * Handles incoming message output items from the backend.
   * Checks for specific structures (like curated config or UE list) and updates messages accordingly.
   * @param {object} message_output - The content of the message output item.
   * @param {Array} prevMessages - The previous state of the messages array.
   * @returns {Array} The new state of the messages array.
   */
  function checkAndHandleMessages(message_output, prevMessages) {
    // Detect curated config structure: has models, network_slice, deployment_location
    if (
      message_output &&
      Array.isArray(message_output.models) &&
      message_output.network_slice &&
      message_output.deployment_location
    ) {
      // Store the latest curated config for access during UE selection/deployment
      setLastCuratedConfig(message_output);
      // Add curated config message type
      return [
        ...prevMessages,
        {
          role: "curated_config",
          content: message_output,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];
    }

    // Determine the message content based on the output structure
    let output =
      typeof message_output == "string"
        ? message_output
        : message_output.questions // Handle cases with 'questions' field
        ? message_output.questions
        : message_output.message; // Handle cases with 'message' field

    if (output) {
      // Add standard assistant message
      return [
        ...prevMessages,
        {
          role: "assistant",
          content: output,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];
    } else {
      // Fallback: log unexpected structure
      console.log("received some other response ", message_output);
      return prevMessages; // Return previous state if output is not recognized
    }
  }

  // Effect hook to process incoming streamed chat events
  useEffect(() => {
    if (!streamedChatEvent) return; // Do nothing if no event

    const event = streamedChatEvent; // Use the event directly
    const eventType = event.event_type;

    // Handle different types of streamed events
    if (eventType === "response_text_delta_event") {
      // Append text delta to the last assistant message
      const response_text_delta = event.response_text_delta;
      setMessages((prevMessages) => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          // Update the last message content
          return [
            ...prevMessages.slice(0, -1),
            {
              ...lastMessage,
              content: lastMessage.content + response_text_delta,
              time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
            },
          ];
        }
        // If last message is not assistant, add a new one
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
      // Add a monotone message for agent activation
      const agent_name = event.agent_name;
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
      // Add a monotone message for tool calls
      const tool_name = event.tool_name;
      const tool_call_item = event.tool_args;
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
      // Add a monotone message for tool outputs
      const tool_output = event.tool_output;
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
      // Handle main message outputs, including UE lists and curated configs
      // const message_output = event.message_output;
      // if (
      //   message_output &&
      //   message_output.isModelRecommendation &&
      //   message_output.ues &&
      //   message_output.ues.length > 0
      // ) {
      //   setMessages((prevMessages) => [
      //     ...prevMessages,
      //     {
      //       role: "ue_selector",
      //       content: message_output.ues, // Pass the UEs array to the selector
      //       time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      //     },
      //   ]);
      //   setChatDisabled(false); // Enable chat after receiving UEs
      //   return; // Stop processing this event further
      // }

      // setMessages((prevMessages) => {
      //   // Remove any "thinking" message before adding the new one
      //   const filtered = prevMessages.filter((msg) => msg.role !== "thinking");
      //   const lastMessage = filtered[filtered.length - 1];

      //   // If it's a curated config, always add a new message
      //   if (
      //     message_output &&
      //     Array.isArray(message_output.models) &&
      //     message_output.network_slice &&
      //     message_output.deployment_location
      //   ) {
      //     return checkAndHandleMessages(message_output, filtered);
      //   }

      //   // If there's no last message or the last message isn't assistant, add a new one
      //   if (!lastMessage || lastMessage.role !== "assistant") {
      //     return checkAndHandleMessages(message_output, filtered);
      //   } else if (lastMessage.content !== message_output) {
      //     // If the content is different, replace the last message
      //     const updatedMessages = [...filtered];
      //     updatedMessages[updatedMessages.length - 1] = {
      //       ...lastMessage,
      //       content: message_output,
      //       time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      //     };
      //     return updatedMessages;
      //   }
      //   // If content is the same, return filtered messages without changes
      //   return filtered;
      // });
      // setChatDisabled(false); // Enable chat after receiving a regular message output
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
      console.log("Unknown event type:", eventType); // Log unknown event types
    }
  }, [streamedChatEvent]); // Re-run effect when streamedChatEvent changes

  // Effect hook for auto-scrolling to the latest message
  useEffect(() => {
    const container = messageContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]); // Re-run effect when messages state changes

  /**
   * Handles sending a message to the backend.
   * Validates input and UE selection (if UESelector is active), constructs chat history,
   * sends the message, and updates the UI.
   */
  const handleSend = () => {
    if (!input.trim()) return; // Do not send empty messages

    // // Check if UESelector was displayed and UEs are selected
    // if (
    //   messages.some((m) => m.role === "ue_selector") &&
    //   selectedUEIMSI.length === 0
    // ) {
    //   alert("Please select at least one UE and press OK before proceeding.");
    //   return;
    // }

    // Construct chat history for the backend
    const chatHistory = [];
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      // Include user, assistant, and monotone messages in history
      if (message.role === "user" || message.role === "assistant") {
        chatHistory.push({
          role: message.role,
          content: message.content,
        });
      } else if (message.role === "monotone") {
        chatHistory.push({
          role: "assistant", // Treat monotone messages as assistant messages in history
          content: `${message.title} ${message.content}`,
        });
      }
      // Note: UESelector and CuratedConfig messages are not included directly in chat history sent back.
    }
    // Add the current user input to the history
    chatHistory.push({
      role: "user",
      content: input,
    });

    // Send the message to the backend
    sendMessage(
      "intelligence_layer", // Target layer
      "network_user_chat", // Target agent/module
      chatHistory
      // {
      //   chat: chatHistory.map(({ role, content }) => ({ role, content })), // Include formatted chat history
      // }
    );

    // Update local state: add user message and thinking indicator
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: input,
        time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      },
      // {
      //   role: "thinking", // Add thinking message immediately after sending
      //   content: "",
      //   time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      //   id: "__thinking__", // Unique ID for easy identification
      // },
    ]);

    setInput(""); // Clear the input field
    setChatDisabled(true); // Disable chat input while waiting for response
  };

  /**
   * Handles key down events in the input textarea.
   * Sends the message on Enter key press (without Shift).
   * @param {object} e - The keyboard event object.
   */
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Prevent default new line
      handleSend(); // Send the message
    }
  };

  // /**
  //  * Renders a single chat message based on its type and content.
  //  * Uses different styles and components for user, assistant, monotone,
  //  * UESelector, CuratedConfig, and Thinking messages.
  //  * @param {object} msg - The message object.
  //  * @param {number} index - The index of the message in the messages array.
  //  * @returns {JSX.Element} The rendered chat message element.
  //  */
  // const renderChatMessage = (msg, index) => {
  //   const isUser = msg.role === "user";
  //   const isAssistant = msg.role === "assistant";
  //   const isMonotone = msg.role === "monotone";
  //   const isCuratedConfig = msg.role === "curated_config";
  //   const isUESelector = msg.role === "ue_selector";
  //   const time = msg.time;

  //   // Render CuratedConfigMessage component
  //   if (isCuratedConfig) {
  //     return (
  //       <div key={index} className="chat chat-start">
  //         <div className="chat-image avatar">
  //           <Image
  //             alt="Agent Icon"
  //             src={agent_icon}
  //             className="w-10 h-10 object-cover"
  //             width={40}
  //             height={40}
  //           />
  //         </div>
  //         <div className="chat-header">
  //           Assistant
  //           <time className="text-xs opacity-50">{time}</time>
  //         </div>
  //         <div
  //           className="chat-bubble chat-bubble-success whitespace-pre-wrap"
  //           style={{ minWidth: 320, maxWidth: 600 }} // Adjust bubble size for content
  //         >
  //           <CuratedConfigMessage
  //             content={msg.content}
  //             chatDisabled={chatDisabled}
  //             ues={selectedUEIMSI} // Pass selected UEs to the component
  //             onDeploy={({
  //               network_slice,
  //               deployment_location,
  //               model,
  //               model_id,
  //               ues,
  //             }) => {
  //               // Back up user's curated selection
  //               setCuratedSelection({
  //                 network_slice,
  //                 deployment_location,
  //                 model,
  //                 model_id,
  //                 ues,
  //               });
  //               setChatDisabled(true); // Disable chat after deployment

  //               // Prepare chat history (following same approach as handleSend)
  //               const chatHistory = [];
  //               for (let i = 0; i < messages.length; i++) {
  //                 const message = messages[i];
  //                 if (["user", "assistant"].includes(message.role)) {
  //                   chatHistory.push({
  //                     role: message.role,
  //                     content: message.content,
  //                   });
  //                 } else if (message.role === "monotone") {
  //                   chatHistory.push({
  //                     role: "assistant",
  //                     content: `${message.title} ${message.content}`,
  //                   });
  //                 }
  //               }
  //               // Add the hardcoded user intent
  //               chatHistory.push({
  //                 role: "user",
  //                 content: `I would like to know the subscriptions that has access to the slice: ${network_slice}`,
  //               });
  //               // Make API call to backend with only chat info for now
  //               sendMessage("intelligence_layer", "network_user_chat", {
  //                 type: "subscription",
  //                 chat: chatHistory,
  //               });
  //             }}
  //           />
  //         </div>
  //       </div>
  //     );
  //   }

  //   // Render ThinkingMessage component
  //   if (msg.role === "thinking") {
  //     return (
  //       <div key={index} className="chat chat-start">
  //         <div className="chat-image avatar">
  //           <Image
  //             alt="Agent Icon"
  //             src={agent_icon}
  //             className="w-10 h-10 object-cover"
  //             width={40}
  //             height={40}
  //           />
  //         </div>
  //         <div className="chat-header">
  //           Assistant
  //           <time className="text-xs opacity-50">{time}</time>
  //         </div>
  //         <div
  //           className="chat-bubble chat-bubble-success whitespace-pre-wrap"
  //           style={{ minWidth: 200, maxWidth: 400 }}
  //         >
  //           <ThinkingMessage /> {/* Render the thinking animation */}
  //         </div>
  //       </div>
  //     );
  //   }

  //   // Render UESelector component
  //   if (isUESelector) {
  //     return (
  //       <div key={index} className="chat chat-start">
  //         <div className="chat-image avatar">
  //           <Image
  //             alt="Agent Icon"
  //             src={agent_icon}
  //             className="w-10 h-10 object-cover"
  //             width={40}
  //             height={40}
  //           />
  //         </div>
  //         <div className="chat-header">
  //           Assistant
  //           <time className="text-xs opacity-50">{time}</time>
  //         </div>
  //         {/* Render the UESelector component inside a chat bubble */}
  //         <div
  //           className="chat-bubble chat-bubble-success"
  //           style={{ minWidth: 240, maxWidth: 560, padding: 0 }}
  //         >
  //           <UESelector
  //             ues={msg.content} // Pass the UEs data to the selector
  //             chatDisabled={chatDisabled}
  //             onSelect={(selectedIMSIs) => {
  //               // Optionally update selection state
  //             }}
  //             onOk={(selectedIMSIs) => {
  //               setSelectedUEIMSI(selectedIMSIs);

  //               // 1. Display the final assistant message
  //               setMessages((prev) => [
  //                 ...prev,
  //                 {
  //                   role: "assistant",
  //                   content:
  //                     "Thank you! The model with your chosen configuration will start deploying soon, and the subscriptions you selected will receive access shortly.",
  //                   time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
  //                 },
  //               ]);

  //               // 2. Send deployment API call if lastCuratedConfig is available
  //               if (lastCuratedConfig) {
  //                 const { models, network_slice, deployment_location } =
  //                   lastCuratedConfig;
  //                 const model = models[0]?.name || "";
  //                 const model_id = models[0]?.id || "";
  //                 sendMessage("intelligence_layer", "network_user_chat", {
  //                   command: "deploy_curated_model_with_subscriptions",
  //                   model,
  //                   deployment: deployment_location,
  //                   slice: network_slice,
  //                   subscriptions: selectedIMSIs,
  //                   model_id,
  //                 });
  //               }

  //               // 3. Permanently disable the chat (cannot send new input)
  //               setChatDisabled(true); // Disable chat for the rest of session
  //             }}
  //           />
  //         </div>
  //       </div>
  //     );
  //   }

  //   // Render standard user, assistant, or monotone messages
  //   return (
  //     <div key={index} className={`chat ${isUser ? "chat-end" : "chat-start"}`}>
  //       {(isAssistant ||
  //         (isMonotone && msg.title && msg.title === "Agent Activated:")) && (
  //         <div className="chat-image avatar">
  //           <Image
  //             alt="Agent Icon"
  //             src={agent_icon}
  //             className="w-10 h-10 object-cover"
  //             width={40}
  //             height={40}
  //           />
  //         </div>
  //       )}
  //       {isMonotone && msg.title && msg.title.includes("Tool") && (
  //         <div className="chat-image avatar">
  //           <Image
  //             alt="Tool Icon"
  //             src={tool_icon}
  //             className="w-10 h-10 object-cover"
  //             width={40}
  //             height={40}
  //           />
  //         </div>
  //       )}
  //       <div className="chat-header">
  //         {isUser ? "User" : isAssistant ? "Assistant" : msg.title}
  //         <time className="text-xs opacity-50">{time}</time>
  //       </div>
  //       <pre
  //         className={`chat-bubble whitespace-pre-wrap ${
  //           isUser
  //             ? "chat-bubble-info"
  //             : isAssistant
  //             ? "chat-bubble-success"
  //             : "bg-base-200 text-base-content"
  //         }`}
  //       >
  //         {msg.content}
  //       </pre>
  //     </div>
  //   );
  // };

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
          onClick={() => {
            if (messages.length > 0) {
              setShowWarning(true);
            } else {
              clearChat();
            }
          }}
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
          disabled={chatDisabled}
        />
        <button
          onClick={handleSend}
          className="btn btn-primary h-full w-20 "
          disabled={chatDisabled}
        >
          <span className="text-xl">SEND</span>
        </button>
      </div>

      {/* Warning Modal */}
      {showWarning && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg">Warning!</h3>
            <p className="py-4">
              Are you sure, your chat history will be deleted?
            </p>
            <div className="modal-action">
              <button
                className="btn btn-ghost"
                onClick={() => {
                  setShowWarning(false);
                  setTempSelectedButton(null);
                }}
              >
                Cancel
              </button>
              <button className="btn btn-primary" onClick={confirmClearChat}>
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
