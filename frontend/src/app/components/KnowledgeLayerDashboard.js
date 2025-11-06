import { useState, useEffect } from "react";

export default function KnowledgeLayerDashboard({
  sendMessage,
  registerMessageHandler,
  deregisterMessageHandler,
}) {
  const [knowledgeQueryInput, setKnowledgeQueryInput] = useState("");
  const [knowledgeLayerRoutes, setKnowledgeLayerRoutes] = useState(null);
  const [knowledgeQueryResponse, setKnowledgeQueryResponse] = useState("");

  useEffect(() => {
    if (!registerMessageHandler) {
      console.error("registerMessageHandler is not provided");
      return;
    }
    registerMessageHandler("knowledge_layer", "get_routes", (response) => {
      console.log("Received knowledge layer routes:", response);
      setKnowledgeLayerRoutes(response);
    });

    registerMessageHandler("knowledge_layer", "query_knowledge", (response) => {
      console.log("Received knowledge query response:", response);
      setKnowledgeQueryResponse(response);
    });

    return () => {
      if (deregisterMessageHandler) {
        deregisterMessageHandler("knowledge_layer", "get_routes");
        deregisterMessageHandler("knowledge_layer", "query_knowledge");
      } else {
        console.error("deregisterMessageHandler is not provided");
      }
    };
  }, []);

  const getKnowledgeRoutes = () => {
    if (!sendMessage) {
      console.error("websocket is not connected");
      return;
    }
    sendMessage("knowledge_layer", "get_routes");
  };

  const generateRouteDotGraph = () => {
    if (!knowledgeLayerRoutes) {
      console.error("Knowledge Layer Routes are not available");
      return;
    }

    const explainerGraphNodes = [];
    const explainerGraphEdges = [];

    knowledgeLayerRoutes.explainer_routes.forEach((explainerRoute) => {
      explainerGraphNodes.push(explainerRoute.pattern);

      explainerRoute.related.forEach((relatedRoute) => {
        explainerGraphEdges.push({
          source: explainerRoute.pattern,
          target: relatedRoute.pattern,
          label: relatedRoute.relationship,
        });
      });
    });

    console.log("Explainer Graph Nodes:", explainerGraphNodes);
    console.log("Explainer Graph Edges:", explainerGraphEdges);

    // Generate DOT language string
    let dot = "digraph KnowledgeRoutes {\n";
    // Add nodes
    explainerGraphNodes.forEach((node) => {
      dot += `  "${node}";\n`;
    });
    // Add edges with labels
    explainerGraphEdges.forEach((edge) => {
      dot += `  "${edge.source}" -> "${edge.target}" [label="${edge.label}"];\n`;
    });
    dot += "}\n";

    // Copy to clipboard
    if (navigator.clipboard) {
      navigator.clipboard
        .writeText(dot)
        .then(() => {
          console.log("DOT graph copied to clipboard!");
        })
        .catch((err) => {
          console.error("Failed to copy DOT graph:", err);
        });
    } else {
      // Fallback for older browsers
      const textarea = document.createElement("textarea");
      textarea.value = dot;
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand("copy");
        console.log("DOT graph copied to clipboard!");
      } catch (err) {
        console.error("Failed to copy DOT graph:", err);
      }
      document.body.removeChild(textarea);
    }
  };

  const queryKnowledge = () => {
    if (!sendMessage) {
      console.error("WebSocket is not connected");
      return;
    }
    if (!knowledgeQueryInput.trim()) {
      console.error("Knowledge query input is empty");
      return;
    }
    sendMessage(
      "knowledge_layer",
      "query_knowledge",
      knowledgeQueryInput.trim()
    );
  };

  useEffect(() => {
    console.log("Knowledge Query Response Updated:", knowledgeQueryResponse);
  }, [knowledgeQueryResponse]);

  return (
    <div className="flex-1 h-full flex flex-col">
      <div className="flex flex-row gap-3 items-center">
        <button
          className="btn btn-outline my-3 mr-3"
          onClick={getKnowledgeRoutes}
        >
          1. Get Knowledge Routes
        </button>
        <button
          className="btn btn-outline my-3 mr-3"
          onClick={generateRouteDotGraph}
          disabled={!knowledgeLayerRoutes}
        >
          2. Generate Route DOT graph
        </button>
        <a
          href="https://dreampuf.github.io/GraphvizOnline/"
          target="_blank"
          className="btn btn-outline my-3"
        >
          3. Open DOT Visualizer Page
        </a>
      </div>

      <div className="flex flex-row gap-3 items-center">
        <input
          className={"input w-[600]"}
          type="text"
          value={knowledgeQueryInput}
          onChange={(e) => setKnowledgeQueryInput(e.target.value)}
          placeholder="Enter your knowledge query key here, e.g., /docs/user_equipments/ue_1/attribute/downlink_cqi"
        />
        <button className="btn btn-outline" onClick={queryKnowledge}>
          Query Knowledge
        </button>
      </div>
      <div className="overflow-y-auto ">
        <div className="my-4">Knowledge Layer Response</div>
        <pre className="whitespace-pre-wrap break-words">
          {knowledgeQueryResponse}
        </pre>
      </div>
    </div>
  );
}
