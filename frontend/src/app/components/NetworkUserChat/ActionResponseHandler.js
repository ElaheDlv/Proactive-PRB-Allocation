import dayjs from "dayjs";

export default function handleUserActionResponse(
  userActionResponse,
  setMessages
) {
  console.log("running handleUserActionResponse with:", userActionResponse);
  if (!userActionResponse) return;

  const actionType = userActionResponse.action_type;

  if ("error" in userActionResponse) {
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        role: "assistant",
        content: `Error: ${userActionResponse.error}`,
        time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
      },
    ]);
    return;
  }

  switch (actionType) {
    case "query_knowledge": {
      const query_response = userActionResponse.query_response;
      console.log("Query response:", query_response);
      console.log("calling setMessages with query response");
      setMessages((prevMessages) => {
        const newMessages = [
          ...prevMessages,
          {
            role: "assistant",
            content: query_response,
            time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
          },
        ];
        console.log("Updated messages:", newMessages);
        return newMessages;
      });
      console.log("setMessages called with query response");
      break;
    }
    default: {
      console.error("Unknown action type:", actionType);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content: `Unknown action type: ${actionType}`,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ]);
      break;
    }
  }
}
