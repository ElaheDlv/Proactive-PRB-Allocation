import dayjs from "dayjs";

export default function handleChatEvent(streamedChatEvent, setMessages) {
  if (!streamedChatEvent) return;

  const event = streamedChatEvent;
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
  } else {
    console.log("Unknown event type:", eventType); // Log unknown event types
  }
}
