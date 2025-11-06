export default function LogDashboard({ simulationState }) {
  if (simulationState === null || simulationState.logs === null) {
    return <div>No simulation logs to show.</div>;
  }

  return (
    <div className="gap-1">
      <div className="log-container">
        {simulationState.logs.map((log, index) => (
          <div key={"log_" + index} className="log">
            {log}
          </div>
        ))}
        {
          simulationState.logs.length === 0 && (
            <div className="log">No logs available.</div>
          )
        }
      </div>
    </div>
  );
}
