import { formatBitrate } from "../utils/formatUtils";
import Image from "next/image";

function renderAIServiceResponse(responseData) {
  if (responseData === null) {
    return <div>No response data available</div>;
  }

  const { ai_service_name, latency, response } = responseData;
  if (!ai_service_name || !latency || !response) {
    return <div>Invalid response data format</div>;
  }

  const error = response["error"];
  const model_response = response["response"];

  if (error) {
    return (
      <div>
        <strong>{ai_service_name}</strong> <br />
        Latency: {latency} ms <br />
        Error: {error} <br />
      </div>
    );
  }

  switch (ai_service_name) {
    case "ultralytics-yolov8-yolov8n": {
      const { model_results, ue_id, visualization } = model_response;
      return (
        <div className="flex flex-row gap-2 items-center">
          {visualization ? (
            <Image
              src={`data:image/png;base64,${visualization}`}
              alt="AI Service Visualization"
              width={320}
              height={240}
              className="my-2 max-w-xs border rounded flex-1"
              unoptimized
            />
          ) : (
            ""
          )}
          <div className="flex-1">
            <strong>AI Service Name:</strong> <br />
            {ai_service_name} <br /> <br />
            <strong>Latency</strong> <br />
            {latency} ms <br /> <br />
            Model Results: <br />
            {JSON.stringify(model_results)}
          </div>

          <br />
        </div>
      );
    }

    case "ultralytics-yolov8-yolov8s": {
      const { model_results, ue_id, visualization } = model_response;
      return (
        <div className="flex flex-row gap-2 items-center">
          {visualization ? (
            <Image
              src={`data:image/png;base64,${visualization}`}
              alt="AI Service Visualization"
              width={320}
              height={240}
              className="my-2 max-w-xs border rounded flex-1"
              unoptimized
            />
          ) : (
            ""
          )}
          <div className="flex-1">
            <strong>AI Service Name:</strong> <br />
            {ai_service_name} <br /> <br />
            <strong>Latency</strong> <br />
            {latency} ms <br /> <br />
            Model Results: <br />
            {JSON.stringify(model_results)}
          </div>

          <br />
        </div>
      );
    }

    default: {
      const { model_results } = model_response;
      return (
        <div>
          <strong>{ai_service_name}</strong> <br />
          Latency: {latency} ms <br />
          Model Results: {JSON.stringify(model_results)} <br />
        </div>
      );
    }
  }

  return (
    <div>
      <strong>{ai_service_name}</strong> <br />
      Latency: {latency} ms <br />
      Response: {JSON.stringify(response)} <br />
    </div>
  );
}

export default function UEDashboard({ simulationState }) {
  if (
    simulationState === null ||
    simulationState.base_station === null ||
    simulationState.UE_list === null
  ) {
    return <div>Simulation Status Not Available Yet.</div>;
  }

  return (
    <div className="gap-1">
      <div className="divider">UE Dashboard</div>
      <div className="overflow-x-auto overflow-y-auto max-h-[1000px]">
        <table className="table table-xs table-pin-rows table-pin-cols">
          <thead>
            <tr>
              <th>UE IMSI</th>
              <td>Position</td>
              <td>QoS</td>
              <td>Channel Condition</td>
              <td>AI Services</td>
            </tr>
          </thead>
          <tbody>
            {simulationState.UE_list.map((ue) => (
              <tr key={ue.ue_imsi}>
                <th className="text-2xl">{ue.ue_imsi}</th>
                <td>
                  <strong>Operation Region</strong> <br />
                  X: {ue.operation_region["min_x"]} -{" "}
                  {ue.operation_region["max_x"]}, Y:{" "}
                  {ue.operation_region["min_y"]} -{" "}
                  {ue.operation_region["max_y"]} <br /> <br />
                  <strong>Current Position</strong> <br />
                  X: {ue.vis_position_x}, Y: {ue.vis_position_y} <br /> <br />
                  <strong>Current Cell</strong> <br />
                  Cell ID: {ue.current_cell}
                </td>
                <td>
                  <strong>Subscribed Slice</strong> <br />
                  Slice: {ue.slice_type} <br /> 5QI: {ue.qos_profile["5QI"]}{" "}
                  <br /> GBR_DL: {formatBitrate(ue.qos_profile["GBR_DL"])}{" "}
                  <br /> GBR_UL: {formatBitrate(ue.qos_profile["GBR_UL"])}
                  <br />
                  <br />
                  <strong>QoS</strong> <br />
                  Achievable Downlink Bitrate:{" "}
                  {formatBitrate(ue.downlink_bitrate)}
                </td>
                <td>
                  <strong>Downlink Signals</strong> <br />
                  {Object.keys(ue.downlink_received_power_dBm_dict).map(
                    (cell_id) => {
                      const frequency_priority =
                        ue.downlink_received_power_dBm_dict[cell_id]
                          .frequency_priority;

                      const received_power_dBm =
                        ue.downlink_received_power_dBm_dict[cell_id]
                          .received_power_dBm;
                      return (
                        <div key={cell_id}>
                          {cell_id}: freq priority: {frequency_priority} signal
                          power: {received_power_dBm.toFixed(2)} dBm
                        </div>
                      );
                    }
                  )}
                  <br />
                  <strong>Downlink SINR / CQI</strong> <br />
                  {ue.downlink_sinr} dB <br /> CQI: {ue.downlink_cqi} <br />{" "}
                  <br />
                  <strong>Downlink MCS Index / MCS Data</strong> <br />
                  MCS Index: {ue.downlink_mcs_index} <br />{" "}
                  {JSON.stringify(ue.downlink_mcs_data)}
                </td>
                <td className="flex flex-col gap-2">
                  {Object.entries(ue.ai_service_responses).map(
                    ([subscription_id, response_data]) => (
                      <div key={subscription_id}>
                        {renderAIServiceResponse(response_data)}
                      </div>
                    )
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
