import React, { useState } from 'react';
import { RepoIcon, BulbIcon } from './Icons'; // Import icons

/**
 * CuratedConfigMessage component: Displays a curated configuration provided by the assistant,
 * allowing the user to select network slice, deployment location, and a specific model,
 * and then deploy the selection.
 * @param {object} content - The curated configuration content (models, network_slice, deployment_location).
 * @param {function} onDeploy - Callback function when the OK button is clicked with the selected configuration.
 * @param {boolean} chatDisabled - Boolean to disable controls after deployment or while chat is globally disabled.
 * @param {Array} ues - Array of selected UE IMSIs to include in the deployment payload.
 */
function CuratedConfigMessage({ content, onDeploy, chatDisabled, ues }) {
  const [networkSlice, setNetworkSlice] = useState(content.network_slice || "");
  const [deploymentLocation, setDeploymentLocation] = useState(content.deployment_location || "");
  const [selectedModel, setSelectedModel] = useState("");
  const [okClicked, setOkClicked] = useState(false); // State to track if OK is clicked to disable controls

  // Define available options for slices and locations
  const networkSliceOptions = ["eMBB", "uRLLC", "mMTC"];
  const deploymentLocationOptions = ["Edge", "cloud"];

  // Filter out duplicate models based on ID
  const uniqueModels = content.models?.filter((model, index, self) =>
    index === self.findIndex((t) => (
      t.id === model.id
    ))
  ) || [];
  const modelOptions = uniqueModels?.map((m) => m.model_name) || [];

  // Determine if the OK button should be enabled
  const canSubmit = networkSlice && deploymentLocation && selectedModel && !okClicked && !chatDisabled;

  // Handler for the OK button click
  const handleOk = () => {
    setOkClicked(true); // Disable controls immediately after clicking OK
    // Find the selected model object from the unique list
    const selectedModelObj = uniqueModels.find(
      (m) => m.model_name === selectedModel
    );

    // Prepare the payload for the deployment API call
    const logPayload = {
      network_slice: networkSlice,
      deployment_location: deploymentLocation,
      model: selectedModel,
      model_id: selectedModelObj ? selectedModelObj.id : undefined, // Include model ID if found
      ues: ues // Include selected UEs
    };

    // Placeholder API call - logs the payload
    console.log("Deploying selection:", logPayload);

    // Call the parent onDeploy handler with the payload
    if (onDeploy) {
      onDeploy(logPayload);
    }
  };

  return (
    <div className="space-y-4">
      <div className="font-semibold">Here is the curated config for your requirement</div>
      {/* Network Slice Selection */}
      <div className="flex flex-col gap-2">
        <label>
          Network Slice:
          <select
            className="select select-bordered ml-2"
            style={{ color: "#fff", backgroundColor: "#222" }}
            value={networkSlice}
            onChange={(e) => setNetworkSlice(e.target.value)}
            disabled={okClicked || chatDisabled} // Disable if OK clicked or chat disabled
          >
            {/* Render network slice options */}
            {networkSliceOptions.map((opt, idx) => (
              <option
                key={opt}
                value={opt}
                style={{ color: "#fff", backgroundColor: "#222" }}
              >
                {`${idx + 1}. ${opt}`}
              </option>
            ))}
          </select>
        </label>
        {/* Deployment Location Selection */}
        <label>
          Deployment Location:
          <select
            className="select select-bordered ml-2"
            style={{ color: "#fff", backgroundColor: "#222" }}
            value={deploymentLocation}
            onChange={(e) => setDeploymentLocation(e.target.value)}
            disabled={okClicked || chatDisabled} // Disable if OK clicked or chat disabled
          >
            {/* Render deployment location options */}
            {deploymentLocationOptions.map((opt, idx) => (
              <option
                key={opt}
                value={opt}
                style={{ color: "#fff", backgroundColor: "#222" }}
              >
                {`${idx + 1}. ${opt}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      {/* Model Selection */}
      <div className="font-semibold mt-2">Here are the models selected for you</div>
      <select
        className="select select-bordered w-full"
        style={{ color: "#fff", backgroundColor: "#222" }}
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        disabled={okClicked || chatDisabled} // Disable if OK clicked or chat disabled
      >
        <option value="" disabled style={{ color: "#bbb", backgroundColor: "#222" }}>
          Select a model
        </option>
        {/* Render model options */}
        {modelOptions.map((model, idx) => (
          <option
            key={model}
            value={model}
            style={{ color: "#fff", backgroundColor: "#222" }}
          >
            {`${idx + 1}. ${model}`}
          </option>
        ))}
      </select>
      {/* OK button */}
      <button
        className="btn btn-primary mt-2"
        onClick={handleOk}
        disabled={!canSubmit} // Enable only when all fields are selected and not already submitted/disabled
      >
        OK
      </button>
      {/* Display selected model details */}
      <div className="mt-4">
        {uniqueModels?.map((model, idx) => (
          <div key={model.id || idx} className="border rounded p-2 mb-2 flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="font-semibold">{`${idx + 1}. ${model.model_name}`}</span>
              {/* Repository link */}
              <a
                href={
                  model.repository_url.startsWith("http://") ||
                  model.repository_url.startsWith("https://")
                    ? model.repository_url
                    : `https://${model.repository_url.replace(/^\/+/, "")}` // Ensure valid URL format
                }
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 text-blue-600 hover:underline flex items-center"
                title="Repository URL"
              >
                <RepoIcon />
                Repo
              </a>
            </div>
            {/* Model rationale/description */}
            <div className="flex items-center gap-2 text-sm text-gray-700">
              <BulbIcon />
              <span>
                {`${idx + 1}. ${model.model_name} -> ${model.rationale}`}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default CuratedConfigMessage;
