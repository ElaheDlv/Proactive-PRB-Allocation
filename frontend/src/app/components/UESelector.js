import React, { useState } from "react";

// Helper function: Groups IMSIs by sorted slice group
function groupUEsBySlices(ues) {
  const groups = {};
  for (const { IMSI, NETWORK_SLICES } of ues) {
    // Sort to form a unique key for the group, e.g., "eMBB,urLLC"
    const key = NETWORK_SLICES.slice().sort().join(", ");
    if (!groups[key]) groups[key] = [];
    groups[key].push(IMSI);
  }
  return groups;
}

/**
 * UESelector component: Allows users to select UEs from a list, grouped by network slices.
 * Provides OK button to confirm selection.
 * @param {Array} ues - Array of UE objects with IMSI and NETWORK_SLICES.
 * @param {function} onSelect - Callback function when selection changes.
 * @param {function} onOk - Callback function when OK button is clicked.
 * @param {boolean} chatDisabled - Boolean to disable controls.
 */
export function UESelector({ ues = [], onSelect, onOk, chatDisabled }) {
  const [selectedIMSI, setSelectedIMSI] = useState(new Set());
  const [warning, setWarning] = useState("");

  if (!ues.length) return <div>No UEs found</div>;

  const groups = groupUEsBySlices(ues);

  // Handler for toggling the selection state of an IMSI
  const toggleIMSI = imsi => {
    setWarning(""); // Clear warning on any selection change
    const newSet = new Set(selectedIMSI);
    if (newSet.has(imsi)) newSet.delete(imsi);
    else newSet.add(imsi);
    setSelectedIMSI(newSet);
    // Call parent onSelect handler with the current selection array
    onSelect && onSelect([...newSet]);
  };

  // Handler for the OK button click
  const handleOk = () => {
    if (selectedIMSI.size < 1) {
      setWarning("Please select at least one UE before proceeding.");
      return;
    }
    // Call parent onOk handler with the final selection array
    onOk && onOk([...selectedIMSI]);
    setWarning(""); // Clear warning after successful OK
  };

  return (
    <div className="bg-base-100 rounded-md p-4" style={{ maxHeight: 400, overflowY: "auto" }}>
      <div className="font-semibold mb-2 text-base-content">Select subscriptions to proceed:</div>
      {/* Render groups of UEs by slice type */}
      {Object.entries(groups).map(([sliceGroup, imsies]) => (
        <div key={sliceGroup} style={{ marginBottom: 12, borderBottom: "1px solid #444" }}> {/* Changed border color */}
          <div className="font-semibold text-sm py-1 text-base-content">{sliceGroup}</div> {/* Adjusted text color */}
          <div className="flex flex-wrap gap-x-3 gap-y-2 pl-2">
            {/* Render individual UE checkboxes */}
            {imsies.map(imsi => (
              <label key={imsi} className="text-base-content" style={{ fontSize: 13, display: "inline-flex", alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={selectedIMSI.has(imsi)}
                  onChange={() => toggleIMSI(imsi)}
                  style={{ marginRight: 5, accentColor: '#4ade80' }}
                  disabled={chatDisabled} // Disable checkbox if chat is disabled
                />
                <span>{imsi}</span>
              </label>
            ))}
          </div>
        </div>
      ))}
      {/* OK button and selection count */}
      <div className="flex gap-3 items-center pt-2">
        <button
          className="btn btn-primary btn-sm"
          style={{ fontSize: 14, padding: '2px 18px' }}
          disabled={chatDisabled} // Disable button if chat is disabled
          onClick={handleOk}
        >
          OK
        </button>
        <span className="text-xs text-base-content">{selectedIMSI.size} selected</span>
      </div>
      {/* Display warning message if any */}
      {warning && <div className="text-xs text-red-500 mt-1">{warning}</div>}
    </div>
  );
}
