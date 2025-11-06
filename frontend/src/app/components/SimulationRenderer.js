import { useEffect, useRef, useCallback, useState } from "react";

export default function SimulationRenderer({ simulationState }) {
  const backgroundCanvasRef = useRef(null);
  const bsCanvasRef = useRef(null);
  const ueCanvasRef = useRef(null);

  // Add state for mouse position
  const [mousePos, setMousePos] = useState(null);

  // Mouse move handler
  const handleMouseMove = (e) => {
    const rect = backgroundCanvasRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * canvasWidth);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * canvasHeight);
    setMousePos({ x, y });
  };

  // Mouse leave handler
  const handleMouseLeave = () => {
    setMousePos(null);
  };

  const renderBackground = () => {
    const canvas = backgroundCanvasRef.current;
    const ctx = canvas.getContext("2d");

    // Set the background color to white
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw the light gray grid
    ctx.strokeStyle = "#D3D3D3";
    ctx.lineWidth = 0.5;

    const gridSize = 20;
    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
  };

  const renderCells = useCallback(() => {
    if (!simulationState || !simulationState.base_stations) {
      return;
    }
    console.log("rendering BS cells ...");
    const canvas = bsCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    simulationState.base_stations.forEach((bsData) => {
      bsData.cell_list.forEach((cellData) => {
        // cellData = {
        //     "frequency_band": "n1",
        //     "carrier_frequency_MHz": 2100,  // in MHz, e.g., from 700 to 28000)
        //     "bandwidth_Hz": 20e6,
        //     "max_prb": 106,
        //     "cell_radius": 300,
        //     "vis_position_x": 200,
        //     "vis_position_y": 200,
        //     "cell_radius": 150,
        // }
        const {
          vis_position_x,
          vis_position_y,
          vis_cell_radius,
          carrier_frequency_MHz,
        } = cellData;
        // Determine the fill color based on the frequency
        let fillColor;
        if (carrier_frequency_MHz < 3000) {
          fillColor = "rgba(135, 206, 250, 0.5)"; // Light blue for low frequencies
        } else if (carrier_frequency_MHz < 20000) {
          fillColor = "rgba(144, 238, 144, 0.5)"; // Light green for mid frequencies
        } else {
          fillColor = "rgba(255, 182, 193, 0.5)"; // Light pink for high frequencies
        }

        // Create a radial gradient
        const gradient = ctx.createRadialGradient(
          vis_position_x,
          vis_position_y,
          0, // Inner circle (center, radius 0)
          vis_position_x,
          vis_position_y,
          vis_cell_radius // Outer circle (center, radius)
        );
        gradient.addColorStop(0, fillColor); // Inner color
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)"); // Outer transparent color

        // Draw the circle with the gradient
        ctx.beginPath();
        ctx.arc(
          vis_position_x,
          vis_position_y,
          vis_cell_radius,
          0,
          2 * Math.PI,
          false
        );
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw the circle's border
        ctx.setLineDash([5, 5]); // Set the dash pattern: 5px dash, 5px gap
        ctx.lineWidth = 1;
        ctx.strokeStyle = "lightgray";
        ctx.stroke();
        ctx.setLineDash([]); // Reset the dash pattern to solid for future drawings
      });

      // Draw the base station's label
      const { bs_id, vis_position_x, vis_position_y } = bsData;
      ctx.font = "14px Arial";
      ctx.fillStyle = "black";
      ctx.textAlign = "center";
      ctx.fillText(`${bs_id}`, vis_position_x, vis_position_y - 10); // Label above the base station
    });
  }, [simulationState]);

  const renderUEs = useCallback(() => {
    if (!simulationState || !simulationState.UE_list) {
      return;
    }

    const canvas = ueCanvasRef.current;
    const ctx = canvas.getContext("2d");

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const ueData of simulationState.UE_list) {
      // Draw the UE as a small 'b' character with the ue id below it
      ctx.font = "20px 'Smooch Sans'";
      ctx.fillStyle = "black";
      ctx.fillText("b", ueData.vis_position_x, ueData.vis_position_y);
      ctx.fillText(
        "[" + ueData.ue_imsi + "]",
        ueData.vis_position_x - ueData.ue_imsi.length * 4,
        ueData.vis_position_y + 16
      ); // Label above the UE

      // Draw solid orange line to the BS that's serving the UE
      if (ueData.current_cell) {
        const servingCell = simulationState.cells.find(
          (cell) => cell.cell_id === ueData.current_cell
        );
        if (servingCell) {
          // Determine the stroke color based on the frequency
          let strokeColor;
          if (ueData.current_cell.includes("low_freq")) {
            strokeColor = "rgba(135, 206, 250, 0.7)"; // Lighter blue for low frequency
          } else if (ueData.current_cell.includes("mid_freq")) {
            strokeColor = "rgba(144, 238, 144, 0.7)"; // Lighter green for mid frequency
          } else if (ueData.current_cell.includes("high_freq")) {
            strokeColor = "rgba(255, 99, 71, 0.7)"; // Lighter red for high frequency
          } else {
            strokeColor = "rgba(169, 169, 169, 0.7)"; // Lighter gray for default
          }

          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(ueData.vis_position_x, ueData.vis_position_y);
          ctx.lineTo(servingCell.vis_position_x, servingCell.vis_position_y);
          ctx.stroke();
        }
      }
    }
  }, [simulationState]);

  useEffect(() => {
    renderBackground();
  }, []);

  useEffect(() => {
    console.log("simulation state changed.");
    console.log(simulationState);
    renderCells();
    renderUEs();
  }, [simulationState, renderCells, renderUEs]);

  const canvasWidth = 1000;
  const canvasHeight = 1000;

  return (
    <div
      className="canvas-container relative my-5"
      style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px` }}
    >
      <canvas
        ref={backgroundCanvasRef}
        width={canvasWidth}
        height={canvasHeight}
        id="background_canvas"
        style={{ position: "absolute", top: 0, left: 0 }}
      />
      <canvas
        ref={bsCanvasRef}
        width={canvasWidth}
        height={canvasHeight}
        id="bs_canvas"
        style={{ position: "absolute", top: 0, left: 0 }}
      />
      <canvas
        ref={ueCanvasRef}
        width={canvasWidth}
        height={canvasHeight}
        id="ue_canvas"
        style={{ position: "absolute", top: 0, left: 0 }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {mousePos && (
        <div
          style={{
            position: "absolute",
            left: mousePos.x,
            top: mousePos.y - 20,
            background: "rgba(0, 0, 255, 0.9)",
            border: "1px solid #ccc",
            borderRadius: "4px",
            padding: "2px 6px",
            fontSize: "12px",
            pointerEvents: "none",
            zIndex: 100,
          }}
        >
          (X: {mousePos.x}, Y: {mousePos.y})
        </div>
      )}
    </div>
  );
}
