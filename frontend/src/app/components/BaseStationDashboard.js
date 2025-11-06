export default function BaseStationDashboard({ simulationState }) {
  if (
    simulationState === null ||
    simulationState.base_station === null ||
    simulationState.UE_list === null
  ) {
    return <div>Simulation Status Not Available Yet.</div>;
  }

  return (
    <div className="gap-1">
      {/* bs_id, cell_id, carrier_frequency_MHz, allocated_prb/max_prb/load, prb_ue_allocation_dict */}
      <div className="grid grid-cols-2 gap-2 px-6">
        {simulationState.base_stations.map((bs) => (
          <div
            key={bs.bs_id + "_bs"}
            className="border-1 border-gray-300 p-2 rounded-md p-4"
          >
            <div className="text-center">Base Station: {bs.bs_id}</div>
            <div className="divider">Edge Server</div>
            <div className="grid grid-cols-2 gap-2 items-center">
              <div>Edge Server ID</div>
              <div>{bs.edge_server.edge_id}</div>
              <div>Node ID</div>
              <div>{bs.edge_server.node_id}</div>
              <div>CPU Memory Available/Total (GB)</div>
              <div>
                {bs.edge_server.available_cpu_memory_GB} /{" "}
                {bs.edge_server.cpu_memory_GB}
              </div>
              <div>Device Type</div>
              <div>{bs.edge_server.device_type}</div>
              <div>Device Memory Available / Total (GB)</div>
              <div>
                {bs.edge_server.available_device_memory_GB} /{" "}
                {bs.edge_server.device_memory_GB}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 items-center">
              {Object.entries(bs.edge_server.ai_service_deployments).map(
                ([ai_service_subscription_id, deployment_data]) => {
                  return (
                    <div
                      key={ai_service_subscription_id}
                      className="m-2 border-1 border-gray-300 p-2 rounded-md p-4"
                    >
                      <div className="text-center">AI Service Deployment</div>
                      <div>&nbsp;</div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>Subscription ID</div>{" "}
                        <div>{ai_service_subscription_id} </div>
                        {/* AI Service Name: <br/>{deployment_data.ai_service_name} <br/><br/> */}
                        <div>AI Service Name</div>
                        <div>{deployment_data.ai_service_name}</div>
                        {/* AI Service Endpoint: <br/>{deployment_data.ai_service_endpoint} <br/><br/> */}
                        <div>AI Service Endpoint</div>
                        <div>{deployment_data.ai_service_endpoint}</div>
                        {/* Image Repository URL: <br/>{deployment_data.image_repository_url} <br/><br/> */}
                        <div>Image Repository URL</div>
                        <div>{deployment_data.image_repository_url}</div>
                        {/* Container Name: <br/>{deployment_data.container_name} <br/><br/> */}
                        <div>Container Name</div>
                        <div>{deployment_data.container_name}</div>
                        {/* Edge CPU Memory Usage (GB): <br/>{deployment_data.edge_specific_cpu_memory_usage_GB} <br/><br/> */}
                        <div>Edge CPU Memory Usage (GB)</div>
                        <div>
                          {deployment_data.edge_specific_cpu_memory_usage_GB}
                        </div>
                        {/* Edge Device Memory Usage (GB): <br/>{deployment_data.edge_specific_device_memory_usage_GB} <br/><br/> */}
                        <div>Edge Device Memory Usage (GB)</div>
                        <div>
                          {deployment_data.edge_specific_device_memory_usage_GB}
                        </div>
                        {/* Countdown Steps: <br/>{deployment_data.countdown_steps} <br/><br/> */}
                        <div>Countdown Steps</div>
                        <div>{deployment_data.countdown_steps}</div>
                        <div>UEs Served</div>
                        <div>{deployment_data.ue_id_list.join(" | ")}</div>
                      </div>
                    </div>
                  );
                }
              )}
            </div>

            {bs.cell_list.map((cell) => (
              <div key={cell.cell_id + "_cell"}>
                <div className="divider">Cell: {cell.cell_id}</div>
                <div className="grid grid-cols-2 gap-2 items-center">
                  <div>Carrier Freq./ BW. </div>
                  <div>
                    {cell.carrier_frequency_MHz} / {cell.bandwidth_Hz / 1e6} MHz
                  </div>
                  <div>
                    Alloc. / Max Downlink PRB <br />
                  </div>
                  <div>
                    {cell.allocated_dl_prb} / {cell.max_dl_prb}
                  </div>
                  <div>
                    Alloc. / Max Uplink PRB <br />
                  </div>
                  <div>
                    {cell.allocated_ul_prb} / {cell.max_ul_prb}
                  </div>
                  <div>Downlink / Up Load</div>
                  <div className="stats">
                    <div className="stat">
                      <div className="stat-value">
                        {(cell.current_dl_load * 100).toFixed(1)} % /
                        {(cell.current_ul_load * 100).toFixed(1)} %
                      </div>
                    </div>
                  </div>
                  <div>Cell Radius</div>
                  <div>{cell.vis_cell_radius * 2} m</div>
                  <div>UE served</div>
                  <div>{Object.keys(cell.prb_ue_allocation_dict).length}</div>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
