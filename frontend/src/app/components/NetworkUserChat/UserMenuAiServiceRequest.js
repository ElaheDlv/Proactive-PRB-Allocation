import { useEffect, useRef, useState } from "react";
import handleChatEvent from "../ChatEventHandler";
import handleUserActionResponse from "./ActionResponseHandler";
import dayjs from "dayjs";

const STEP_SERVICE_NEED_PROFILING = "step_service_need_profiling";
const STEP_SERVICE_DEPLOYMENT = "step_service_deployment";


export default function UserMenuAIServiceRequest({
  registerMessageHandler,
  deregisterMessageHandler,
  sendMessage,
  setMessages,
  setCurrentMenu,
}) {
  const [optionButtonList, setOptionButtonList] = useState([]);
  const [chatDisabled, setChatDisabled] = useState(false);
  const [input, setInput] = useState("");
  const messageSent = useRef(false);
  const currentStep = useRef(null);

  // this is to handle the strict mode of React double rendering.
  const initialMessageAdded = useRef(false);

  const getDefaultOptionButtonList = () => [
    {
      label: "RESET",
      onClick: () => setCurrentMenu("main_menu"),
    },
    {
      label:
        "Test: I'm working on a fleed of autonomous ground vehicles to search stray animals in my university campus.",
      onClick: () =>
        onSendQuickMessage(
          STEP_SERVICE_NEED_PROFILING,
          "I'm working on a fleet of autonomous ground vehicles to search stray animals in my university campus."
        ),
    },
    {
      label:
        "Test: yeah the vehicles are equipped with wide angle cameras. I wish to be able to recognize different animal species, such that I don't miss dogs or cats but can safely ignore birds.",
      onClick: () =>
        onSendQuickMessage(
          STEP_SERVICE_NEED_PROFILING,
          "yeah the vehicles are equipped with wide angle cameras. I wish to be able to recognize different animal species, such that I don't miss dogs or cats but can safely ignore birds."
        ),
    },
    {
      label: "Test: Select AI service: ultralytics-yolov8-yolov8n",
      onClick: () => {
        onSelectAIService("ultralytics-yolov8-yolov8n");
      },
    },
    {
      label: "Test: Please recommend.",
      onClick: () => {
        onSendQuickMessage(
          STEP_SERVICE_NEED_PROFILING,
          "Please recommend some AI services."
        );
      },
    },
    {
      label: "Test: IMSI_1, IMSI_2",
      onClick: () => {
        onSendQuickMessage(STEP_SERVICE_DEPLOYMENT, "IMSI_1, IMSI_2");
      },
    },
    {
      label: "Test: Deploy sample AI service.",
      onClick: onDeployTestAIService,
    },
  ];

  const initMenu = () => {
    console.log("Initializing AI Service Request Menu.");
    if (!initialMessageAdded.current) {
      setMessages((prevMessages) => {
        console.log("set messages called: ");
        console.log(prevMessages);
        return [
          ...prevMessages,
          {
            role: "user",
            content: "I wish to request an AI service for my use case.",
            time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
          },
        ];
      });
      initialMessageAdded.current = true;
    }
    setOptionButtonList(getDefaultOptionButtonList());
    setCurrentStep(STEP_SERVICE_NEED_PROFILING);
  };

  const onSelectAIService = (aiServiceName) => {
    console.log(`Selected AI Service: ${aiServiceName}`);
    messageSent.current = false; // Reset message sent flag
    setCurrentStep(STEP_SERVICE_DEPLOYMENT);
    setMessages((prevMessages) => {
      const newMessages = [
        ...prevMessages,
        {
          role: "user",
          content: `I would like to deploy the AI service: ${aiServiceName}.`,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];

      if (!messageSent.current) {
        // Send the message to the backend
        setChatDisabled(true); // Disable chat while processing
        sendMessage("intelligence_layer", "ai_service_pipeline", {
          current_step: STEP_SERVICE_DEPLOYMENT,
          ai_service_name: aiServiceName,
          messages: newMessages.map((msg) => {
            if (msg.role !== "monotone") {
              return {
                role: msg.role,
                content: msg.content,
              };
            } else {
              return {
                role: "assistant", // Convert monotone messages to assistant role
                content: msg.title + "\n" + msg.content,
              };
            }
          }),
        });
        messageSent.current = true; // Set flag to true after sending the message
      }

      return newMessages;
    });
  };

  const onSendQuickMessage = (current_step, messageContent) => {
    if (current_step) {
      // when the new step is provided, update it, otherwise use the current step
      setCurrentStep(current_step);
    }

    messageSent.current = false; // Reset message sent flag
    setMessages((prevMessages) => {
      const newMessages = [
        ...prevMessages,
        {
          role: "user",
          content: messageContent,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];
      // the setMessage callback is called twice in React strict mode
      if (!messageSent.current) {
        setChatDisabled(true); // Disable chat while processing
        sendMessage("intelligence_layer", "ai_service_pipeline", {
          current_step: current_step,
          messages: newMessages.map((msg) => {
            if (msg.role !== "monotone") {
              return {
                role: msg.role,
                content: msg.content,
              };
            } else {
              return {
                role: "assistant", // Convert monotone messages to assistant role
                content: msg.title + "\n" + msg.content,
              };
            }
          }),
        });
        messageSent.current = true; // Set flag to true after sending the message
      }
      return newMessages;
    });
  };

  const onDeployTestAIService = () => {
    messageSent.current = false; // Reset message sent flag
    setMessages((prevMessages) => {
      const newMessages = [
        ...prevMessages,
        {
          role: "user",
          content:
            "Deploying a test AI service: ultralytics-yolov8-yolov8n for UE IMSI_1.",
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];
      // the setMessage callback is called twice in React strict mode
      if (!messageSent.current) {
        setChatDisabled(true); // Disable chat while processing
        sendMessage("intelligence_layer", "ai_service_pipeline", {
          current_step: "step_test_ai_service_deployment",
          data: {
            ai_service_name: "ultralytics-yolov8-yolov8n",
            ue_id_list: ["IMSI_1"],
          },
          messages: newMessages.map((msg) => {
            if (msg.role !== "monotone") {
              return {
                role: msg.role,
                content: msg.content,
              };
            } else {
              return {
                role: "assistant", // Convert monotone messages to assistant role
                content: msg.title + "\n" + msg.content,
              };
            }
          }),
        });
        messageSent.current = true; // Set flag to true after sending the message
      }
      return newMessages;
    });
  };

  const streamedChatEventHandler = (streamedChatEvent) => {
    if (!streamedChatEvent) return;
    const eventType = streamedChatEvent.event_type;

    if (eventType === "ai_services_recommendation") {
      const ai_service_names = streamedChatEvent.ai_service_names;
      const ai_service_descriptions = streamedChatEvent.ai_service_descriptions;

      if (!ai_service_names || !ai_service_descriptions) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            role: "assistant",
            content: "No AI services available at the moment.",
            time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
          },
        ]);
        setOptionButtonList(getDefaultOptionButtonList());
        return;
      }

      setOptionButtonList([
        ...getDefaultOptionButtonList(),
        ...ai_service_names.map((name, index) => ({
          label: `Select recommended AI Service: ${name}`,
          onClick: () => onSelectAIService(name),
        })),
        {
          label: "Show me other AI service candidates please.",
          onClick: () => {
            onSendQuickMessage(
              STEP_SERVICE_NEED_PROFILING,
              "Show me other AI service candidates please."
            );
            setOptionButtonList(getDefaultOptionButtonList());
          },
        },
      ]);

      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content: `Recommended AI Services: 

${ai_service_descriptions.join("\n\n")}

Please select one of the above AI serivces that you would like to deploy.`,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ]);

      return;
    }

    if (eventType === "ai_service_deployment") {
      const message = streamedChatEvent.message;
      if (message) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            role: "assistant",
            content: message,
            time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
          },
        ]);
        setOptionButtonList(getDefaultOptionButtonList());
        return;
      }
    }

    handleChatEvent(streamedChatEvent, setMessages);
    setChatDisabled(false); // Re-enable chat after processing the event
  };

  const UserActionResponseHandler = (userActionResponse) => {
    handleUserActionResponse(userActionResponse, setMessages);
  };

  useEffect(() => {
    if (registerMessageHandler) {
      registerMessageHandler(
        "intelligence_layer",
        "ai_service_pipeline_response",
        streamedChatEventHandler
      );

      registerMessageHandler(
        "intelligence_layer",
        "network_user_action_response",
        UserActionResponseHandler
      );
    }

    initMenu();

    return () => {
      if (deregisterMessageHandler) {
        deregisterMessageHandler(
          "intelligence_layer",
          "ai_service_pipeline_response"
        );
        deregisterMessageHandler(
          "intelligence_layer",
          "network_user_action_response"
        );
      }
      console.log("UserMenuAIServiceRequest cleanup done.");
    };
  }, []);

  const setCurrentStep = (newStep) => {
    if (currentStep.current === newStep) {
      return;
    }

    currentStep.current = newStep;
    console.log(`Current step updated to: ${newStep}`);

    if (!currentStep.current) return;

    if (currentStep.current === STEP_SERVICE_NEED_PROFILING) {
      setMessages((prevMessages) => {
        console.log("set messages called: ");
        console.log(prevMessages);
        return [
          ...prevMessages,
          {
            role: "monotone",
            title: "Monotone",
            content: `Please describe the AI service you need or the use case you are working on.`,
            time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
          },
        ];
      });
    }
  };

  const handleSend = () => {
    messageSent.current = false; // Reset message sent flag
    setMessages((prevMessages) => {
      const updatedMessages = [
        ...prevMessages,
        {
          role: "user",
          content: input,
          time: dayjs().format("{YYYY} MM-DDTHH:mm:ss SSS [Z] A"),
        },
      ];
      if (!messageSent.current) {
        setChatDisabled(true); // Disable chat while processing
        sendMessage("intelligence_layer", "ai_service_pipeline", {
          current_step: currentStep.current,
          messages: updatedMessages.map((msg) => {
            if (msg.role !== "monotone") {
              return {
                role: msg.role,
                content: msg.content,
              };
            } else {
              return {
                role: "assistant", // Convert monotone messages to assistant role
                content: msg.content,
              };
            }
          }),
        });
        messageSent.current = true; // Set flag to true after sending the message
      }

      return updatedMessages;
    });

    setInput("");
    // setChatDisabled(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() === "") return;
      handleSend();
      return;
    }
    if (e.key === "Escape") {
      setInput("");
    }
  };

  return (
    <>
      {/* Options */}
      <div className="p-4 border-t border-base-300 bg-base-100">
        <div className="flex flex-row gap-2 items-center flex-wrap">
          <span>OPTIONS</span>
          {optionButtonList.map((option, index) => (
            <button
              key={index}
              onClick={option.onClick}
              className="btn btn-info"
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* Input Box */}
      <div className="p-4 border-t border-base-300 bg-base-100 flex items-center gap-2">
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
    </>
  );
}
