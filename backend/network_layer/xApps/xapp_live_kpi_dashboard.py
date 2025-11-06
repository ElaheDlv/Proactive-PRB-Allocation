from .xapp_base import xAppBase

import threading
from collections import defaultdict, deque

# Dash / Plotly
import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go

from settings import (
    RAN_PRB_CAP_SLIDER_DEFAULT,
    RAN_PRB_CAP_SLIDER_MAX,
    RAN_SLICE_DL_PRB_SPLIT_DEFAULT,
    RAN_SLICE_DL_PRB_OVERRIDES,
    RAN_SLICE_KNOB_STEP_FRAC,
    RAN_KPI_MAX_POINTS,
    RAN_KPI_LOG_ENABLE,
    RAN_KPI_LOG_DIR,
    RAN_KPI_HISTORY_ENABLE,
)
import settings

import os
import csv
from datetime import datetime

# Rolling window sizing and refresh
MAX_POINTS = RAN_KPI_MAX_POINTS
REFRESH_SEC = 0.5
DASH_PORT = int(os.getenv("DASH_PORT", "8059"))
SLICE_MOVE_STEP_PRBS = 1  # Treat one RB (paper) as 1 PRBs here


# --- Layout helpers (keeps graphs compact) ---
SLICE_COLORS = {
    "eMBB": "#1f77b4",   # blue
    "URLLC": "#d62728",  # red
    "mMTC": "#2ca02c",   # green
    None: "#7f7f7f",       # grey fallback
}
CONTAINER_STYLE = {
    "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    "padding": "12px",
    "maxWidth": "1400px",
    "margin": "0 auto",
}

ROW_2COL = {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"}
ROW_1COL = {"display": "grid", "gridTemplateColumns": "1fr", "gap": "12px"}


def tidy(fig, title, ytitle, show_legend=True, overlay_legend=False, x_title="Trace time (s)"):
    """Apply common layout with optional legend overlay to save vertical space.

    - When overlay_legend is True, place the legend inside the plotting area at
      the top-left with semi‑transparent background so it does not shrink the plot.
    - When False, place the legend above (if history slider enabled) or below.
    """
    base_margin = dict(l=40, r=10, t=40, b=35)
    legend_layout = dict(orientation="h", y=-0.25, x=0)

    if overlay_legend:
        # Overlay legend inside the plot; keep compact font and background
        legend_layout = dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            x=0.01,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=9),
        )
    else:
        # Non-overlay legends reserve space in margins
        if RAN_KPI_HISTORY_ENABLE:
            base_margin = dict(l=50, r=20, t=70, b=80)
            legend_layout = dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom")

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=ytitle,
        height=320,
        margin=base_margin,
        legend=legend_layout,
        showlegend=bool(show_legend),
        template="plotly_white",
        uirevision="kpi-static",
    )
    if RAN_KPI_HISTORY_ENABLE:
        try:
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.12))
        except Exception:
            pass
    return fig


def _deque():
    # If MAX_POINTS <= 0, keep unbounded history
    maxlen = None if (MAX_POINTS is None or MAX_POINTS <= 0) else int(MAX_POINTS)
    return deque(maxlen=maxlen)


class xAppLiveKPIDashboard(xAppBase):
    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = True

        # Rolling time axis (simulation step)
        self._t = _deque()

        # --- Per‑UE series ---
        self._ue_dl_mbps = defaultdict(_deque)      # {IMSI: deque}
        self._ue_sinr_db = defaultdict(_deque)      # {IMSI: deque}
        self._ue_cqi     = defaultdict(_deque)      # {IMSI: deque}
        self._ue_dl_buf  = defaultdict(_deque)      # optional: {IMSI: deque}
        self._ue_dl_prb  = defaultdict(_deque)      # from cell allocation map if present
        self._ue_dl_prb_req = defaultdict(_deque)   # requested PRBs per UE if exposed by Cell
        self._ue_dl_latency = defaultdict(_deque)  # {IMSI: deque}
        
        # --- Per‑Cell series ---
        self._cell_dl_load    = defaultdict(_deque)  # {cell_id: deque in [0,1]}
        self._cell_alloc_prb  = defaultdict(_deque)  # {cell_id: deque}
        self._cell_max_prb    = defaultdict(_deque)  # {cell_id: deque}

        # Concurrency
        self._lock = threading.Lock()

        # Dash server thread
        self._dash_app = None
        self._dash_thread = None

        # Last step seen (avoid double pushes)
        self._last_step = None

        # Live control (per‑UE PRB cap)
        self._prb_cap = None

        # KPI logging
        self._log_enabled = RAN_KPI_LOG_ENABLE
        self._log_dir = RAN_KPI_LOG_DIR
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ue_log_fp = None
        self._cell_log_fp = None
        self._ue_log_csv = None
        self._cell_log_csv = None

        # View controls: legend and UE filtering
        self._show_legend = False  # default to no built-in legend
        self._overlay_legend = False
        self._ue_filter = set()  # empty -> show all

    # ---------------- xApp lifecycle ----------------
    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        # Initialize logging if enabled
        if self._log_enabled:
            try:
                os.makedirs(self._log_dir, exist_ok=True)
                ue_path = os.path.join(self._log_dir, f"ue_kpis_{self._run_id}.csv")
                cell_path = os.path.join(self._log_dir, f"cell_kpis_{self._run_id}.csv")
                self._ue_log_fp = open(ue_path, "w", newline="")
                self._cell_log_fp = open(cell_path, "w", newline="")
                self._ue_log_csv = csv.DictWriter(
                    self._ue_log_fp,
                    fieldnames=[
                        "sim_step",
                        "imsi",
                        "dl_bps",
                        "dl_mbps",
                        "sinr_db",
                        "cqi",
                        "dl_buffer_bytes",
                        "dl_prb_granted",
                        "dl_prb_requested",
                        "dl_latency_ms",
                    ],
                )
                self._ue_log_csv.writeheader()
                self._cell_log_csv = csv.DictWriter(
                    self._cell_log_fp,
                    fieldnames=[
                        "sim_step",
                        "cell_id",
                        "dl_load",
                        "allocated_prb",
                        "max_prb",
                    ],
                )
                self._cell_log_csv.writeheader()
                print(
                    f"{self.xapp_id}: KPI logging enabled. UE log: {ue_path} | Cell log: {cell_path}"
                )
            except Exception as e:
                print(f"{self.xapp_id}: Failed to initialize KPI logging: {e}")
                self._log_enabled = False
        self._start_dashboard()

    def _default_slice_fraction(self, slice_name: str) -> float:
        """Return initial slider fraction honouring absolute PRB overrides when present."""
        try:
            overrides = RAN_SLICE_DL_PRB_OVERRIDES or {}
        except Exception:
            overrides = {}
        if overrides:
            max_prb = None
            try:
                first_cell = next(iter(self.cell_list.values())) if self.cell_list else None
                max_prb = getattr(first_cell, "max_dl_prb", None)
            except Exception:
                max_prb = None
            if max_prb:
                try:
                    quota = float(overrides.get(slice_name, 0))
                    return max(0.0, min(1.0, quota / float(max_prb)))
                except Exception:
                    pass
        return float(RAN_SLICE_DL_PRB_SPLIT_DEFAULT.get(slice_name, 0.0))

    def step(self):
        """Collect KPIs each simulation step."""
        sim_step = getattr(getattr(self.ric, "simulation_engine", None), "sim_step", None)
        if sim_step is None or sim_step == self._last_step:
            return
        self._last_step = sim_step

        with self._lock:
            self._t.append(sim_step)

            # ---- Per‑UE ----
            for imsi, ue in self.ue_list.items():
                # DL bitrate is stored in bps on the UE; convert to Mbps for plotting
                dl_bps = float(getattr(ue, "downlink_bitrate", 0.0) or 0.0)
                self._ue_dl_mbps[imsi].append(dl_bps / 1e6)

                # SINR (dB)
                sinr = getattr(ue, "downlink_sinr", None)
                if sinr is not None:
                    self._ue_sinr_db[imsi].append(float(sinr))

                # CQI
                cqi = getattr(ue, "downlink_cqi", None)
                if cqi is not None:
                    self._ue_cqi[imsi].append(float(cqi))

                # Optional queues/buffers (if UE defines them)
                if hasattr(ue, "dl_buffer_bytes"):
                    buf = float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
                    self._ue_dl_buf[imsi].append(buf)


                # Downlink latency (if UE exposes it)
                dl_latency = float(getattr(ue, "downlink_latency", 0.0) or 0.0)
                self._ue_dl_latency[imsi].append(dl_latency)
                
                
                # Allocated PRBs for this UE (DL)
                cell = getattr(ue, "current_cell", None)
                dl_prb = None
                dl_requested = None
                if cell is not None:
                    alloc_map = getattr(cell, "prb_ue_allocation_dict", {}) or {}
                    ue_alloc = alloc_map.get(imsi, {})
                    dl_prb = ue_alloc.get("downlink", None)
                    if dl_prb is not None:
                        self._ue_dl_prb[imsi].append(float(dl_prb))

                    # Requested PRBs (if Cell exposes per‑UE demand map)
                    dl_req_map = getattr(cell, "dl_total_prb_demand", {}) or {}
                    dl_requested = dl_req_map.get(imsi, None)
                    if dl_requested is not None:
                        self._ue_dl_prb_req[imsi].append(float(dl_requested))

                # Logging per‑UE KPIs (one row per UE per step)
                if self._log_enabled and self._ue_log_csv is not None:
                    try:
                        self._ue_log_csv.writerow(
                            {
                                "sim_step": sim_step,
                                "imsi": imsi,
                                "dl_bps": dl_bps,
                                "dl_mbps": dl_bps / 1e6,
                                "sinr_db": float(getattr(ue, "downlink_sinr", 0.0) or 0.0),
                                "cqi": float(getattr(ue, "downlink_cqi", 0.0) or 0.0),
                                "dl_buffer_bytes": float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0),
                                "dl_prb_granted": float(dl_prb) if dl_prb is not None else "",
                                "dl_prb_requested": float(dl_requested) if dl_requested is not None else "",
                                "dl_latency_ms": dl_latency * 1000.0,
                            }
                        )
                    except Exception:
                        pass
                        



            # ---- Per‑Cell ----
            for cell_id, cell in self.cell_list.items():
                load = getattr(cell, "current_dl_load", None)
                if load is not None:
                    self._cell_dl_load[cell_id].append(float(load))

                alloc_dl = getattr(cell, "allocated_dl_prb", None)
                if alloc_dl is not None:
                    self._cell_alloc_prb[cell_id].append(float(alloc_dl))

                max_prb = getattr(cell, "max_dl_prb", None)
                if max_prb is not None:
                    self._cell_max_prb[cell_id].append(float(max_prb))

                # Logging per‑cell KPIs (one row per cell per step)
                if self._log_enabled and self._cell_log_csv is not None:
                    try:
                        self._cell_log_csv.writerow(
                            {
                                "sim_step": sim_step,
                                "cell_id": cell_id,
                                "dl_load": float(load) if load is not None else "",
                                "allocated_prb": float(alloc_dl) if alloc_dl is not None else "",
                                "max_prb": float(max_prb) if max_prb is not None else "",
                            }
                        )
                    except Exception:
                        pass

            # Flush logs to disk each step (small runs; fine for now)
            if self._log_enabled:
                try:
                    if self._ue_log_fp:
                        self._ue_log_fp.flush()
                    if self._cell_log_fp:
                        self._cell_log_fp.flush()
                except Exception:
                    pass

    # ---------------- Dash server ----------------
    def _start_dashboard(self):
        if self._dash_thread and self._dash_thread.is_alive():
            return

        app = Dash(__name__)
        self._dash_app = app

        app.layout = html.Div(
            style=CONTAINER_STYLE,
            children=[
                html.H2("Live RAN KPI Dashboard"),
                html.P("Streaming KPIs directly from UEs/Cells within the simulator."),

                # Controls
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "8px"},
                    children=[
                        html.Div([
                            html.Label("Max DL PRBs per UE (live)"),
                            dcc.Slider(
                                id="prb-cap",
                                min=0,
                                max=RAN_PRB_CAP_SLIDER_MAX,
                                step=1,
                                value=(RAN_PRB_CAP_SLIDER_DEFAULT or 0),
                                tooltip={"always_visible": False},
                                marks={0: "0", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50", RAN_PRB_CAP_SLIDER_MAX: str(RAN_PRB_CAP_SLIDER_MAX)},
                            ),
                            html.Small("Set lower to throttle any single UE (0 = unlimited)."),
                        ]),
                        html.Div(id="prb-cap-label", style={"alignSelf": "center"}),
                    ],
                ),

                # Display options: external legend (scrollable) and toggles
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginTop": "8px"},
                    children=[
                        html.Div([
                            html.Label("UE Legend (scroll)", title="Toggle UEs to display in plots"),
                            html.Div([
                                dcc.Checklist(id="ue-list", options=[], value=[], labelStyle={"display": "block", "cursor": "pointer"}),
                            ], style={"maxHeight": "160px", "overflowY": "auto", "border": "1px solid #ddd", "padding": "6px", "backgroundColor": "#fafafa"}),
                            html.Div([
                                html.Button("Select All", id="btn-ue-select-all", n_clicks=0, style={"marginRight": "6px"}),
                                html.Button("Clear", id="btn-ue-clear", n_clicks=0),
                            ], style={"marginTop": "6px"}),
                        ]),
                        html.Div([
                            html.Label("Legend visibility"),
                            dcc.Checklist(
                                id="legend-toggle",
                                options=[{"label": "Show built‑in legend", "value": "show"}],
                                value=[],
                            ),
                        ]),
                        html.Div([
                            html.Label("Legend placement"),
                            dcc.Checklist(
                                id="legend-overlay",
                                options=[{"label": "Overlay legend inside plot", "value": "overlay"}],
                                value=[],
                            ),
                        ]),
                    ],
                ),
                html.Div(id="view-controls-label", style={"marginTop": "4px", "fontSize": "12px", "color": "#666"}),

                # Live per-cell PRB quotas (updated each tick)
                html.Div(id="cell-quota-live", style={"marginTop": "6px", "fontSize": "12px", "color": "#333"}),

                # Slice split controls (fractions per cell)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginTop": "8px"},
                    children=[
                        html.Div([
                            html.Label("eMBB share (0-1)"),
                            dcc.Slider(
                                id="slice-embb-share",
                                min=0,
                                max=1.0,
                                step=RAN_SLICE_KNOB_STEP_FRAC,
                                value=self._default_slice_fraction("eMBB"),
                                tooltip={"always_visible": False},
                            ),
                        ]),
                        html.Div([
                            html.Label("URLLC share (0-1)"),
                            dcc.Slider(
                                id="slice-urllc-share",
                                min=0,
                                max=1.0,
                                step=RAN_SLICE_KNOB_STEP_FRAC,
                                value=self._default_slice_fraction("URLLC"),
                                tooltip={"always_visible": False},
                            ),
                        ]),
                        html.Div([
                            html.Label("mMTC share (0-1)"),
                            dcc.Slider(
                                id="slice-mmtc-share",
                                min=0,
                                max=1.0,
                                step=RAN_SLICE_KNOB_STEP_FRAC,
                                value=self._default_slice_fraction("mMTC"),
                                tooltip={"always_visible": False},
                            ),
                        ]),
                    ],
                ),
                html.Div(id="slice-share-label", style={"marginTop": "4px"}),

                # Discrete RB move controls (Table 3 actions)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "8px", "marginTop": "12px"},
                    children=[
                        dcc.Dropdown(
                            id="cell-selector",
                            options=(
                                [{"label": "ALL", "value": "ALL"}]
                                + [{"label": cid, "value": cid} for cid in sorted(self.cell_list.keys())]
                            ),
                            value="ALL",
                            clearable=False,
                        ),
                        html.Button("Keep", id="act-0", n_clicks=0),
                        html.Button("mMTC→URLLC", id="act-1", n_clicks=0),
                        html.Button("mMTC→eMBB", id="act-2", n_clicks=0),
                        html.Button("URLLC→mMTC", id="act-3", n_clicks=0),
                        html.Button("URLLC→eMBB", id="act-4", n_clicks=0),
                        html.Button("eMBB→mMTC", id="act-5", n_clicks=0),
                        html.Button("eMBB→URLLC", id="act-6", n_clicks=0),
                    ],
                ),
                html.Div(id="slice-move-label", style={"marginTop": "4px"}),
                html.Div(id="ue-slice-list", style={"marginTop": "4px", "fontSize": "12px", "color": "#444"}),

                html.Hr(),

                html.Div(style=ROW_1COL, children=[dcc.Graph(id="ue-bitrate")]),
                # Move PRB plots just below bitrate for quick visibility
                html.Div(style=ROW_2COL, children=[
                    dcc.Graph(id="ue-prb-granted"),
                    dcc.Graph(id="ue-prb-requested"),
                ]),
                # Place SINR/CQI below PRB plots
                html.Div(style=ROW_2COL, children=[
                    dcc.Graph(id="ue-sinr"),
                    dcc.Graph(id="ue-cqi"),
                ]),
                html.Div(style=ROW_1COL, children=[dcc.Graph(id="cell-load")]),
                html.Div(style=ROW_1COL, children=[dcc.Graph(id="ue-buffer")]),
                
                html.Div(style=ROW_1COL, children=[dcc.Graph(id="ue-dl-latency")]),

                dcc.Interval(id="tick", interval=int(REFRESH_SEC * 1000), n_intervals=0),
            ],
        )

        @app.callback(Output("prb-cap-label", "children"), Input("prb-cap", "value"))
        def _set_cap(val):
            with self._lock:
                # 0/unlimited handling: 0 means unlimited; otherwise int cap
                self._prb_cap = int(val) if val is not None else 0
                cap_to_apply = None if self._prb_cap == 0 else self._prb_cap
                for cell in self.cell_list.values():
                    setattr(cell, "prb_per_ue_cap", cap_to_apply)
                return f"Current cap: {cap_to_apply if cap_to_apply is not None else 'unlimited'} PRBs/UE"

        @app.callback(
            Output("view-controls-label", "children"),
            Input("ue-list", "value"),
            Input("legend-toggle", "value"),
            Input("legend-overlay", "value"),
        )
        def _view_controls(ue_values, show_values, overlay_values):
            with self._lock:
                sels = set(ue_values or [])
                self._ue_filter = set() if not sels else set(sels)
                self._show_legend = (show_values is not None) and ("show" in show_values)
                self._overlay_legend = (overlay_values is not None) and ("overlay" in overlay_values)
            # Reuse this label to acknowledge controls changed
            n = "ALL" if not self._ue_filter else f"{len(self._ue_filter)} selected"
            mode = "overlay" if self._overlay_legend else "separate"
            vis = "shown" if self._show_legend else "hidden"
            return f"UE filter: {n}. Legend: {vis}, {mode}."

        @app.callback(
            Output("slice-share-label", "children"),
            Input("slice-embb-share", "value"),
            Input("slice-urllc-share", "value"),
            Input("slice-mmtc-share", "value"),
        )
        def _set_slice_shares(v_embb, v_urllc, v_mmtc):
            # Normalize if sum > 1.0
            shares = {
                "eMBB": float(v_embb or 0.0),
                "URLLC": float(v_urllc or 0.0),
                "mMTC": float(v_mmtc or 0.0),
            }
            ssum = sum(shares.values())
            if ssum > 1.0 and ssum > 0:
                scale = 1.0 / ssum
                for k in shares:
                    shares[k] = round(shares[k] * scale, 4)
            with self._lock:
                for cell in self.cell_list.values():
                    if hasattr(cell, "set_slice_quota_by_fraction"):
                        cell.set_slice_quota_by_fraction(shares)
            # Preview for one arbitrary cell
            any_cell = next(iter(self.cell_list.values())) if self.cell_list else None
            if any_cell:
                quotas = {k: int(any_cell.max_dl_prb * shares.get(k, 0.0)) for k in shares}
                return (
                    f"Slice shares set to eMBB={shares['eMBB']:.2f}, URLLC={shares['URLLC']:.2f}, "
                    f"mMTC={shares['mMTC']:.2f}. Example quotas on {any_cell.cell_id}: {quotas} "
                    f"(of max {any_cell.max_dl_prb})."
                )
            return (
                f"Slice shares set to eMBB={shares['eMBB']:.2f}, URLLC={shares['URLLC']:.2f}, "
                f"mMTC={shares['mMTC']:.2f}."
            )

        @app.callback(
            Output("slice-move-label", "children"),
            Input("cell-selector", "value"),
            Input("act-0", "n_clicks"),
            Input("act-1", "n_clicks"),
            Input("act-2", "n_clicks"),
            Input("act-3", "n_clicks"),
            Input("act-4", "n_clicks"),
            Input("act-5", "n_clicks"),
            Input("act-6", "n_clicks"),
            prevent_initial_call=True,
        )
        def _handle_move(cell_sel, a0, a1, a2, a3, a4, a5, a6):
            # Determine which button triggered
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            trig = ctx.triggered[0]["prop_id"].split(".")[0]
            if trig == "act-0":
                return "No change applied."
            # Action mapping: (src, dst)
            action_map = {
                "act-1": ("mMTC", "URLLC"),
                "act-2": ("mMTC", "eMBB"),
                "act-3": ("URLLC", "mMTC"),
                "act-4": ("URLLC", "eMBB"),
                "act-5": ("eMBB", "mMTC"),
                "act-6": ("eMBB", "URLLC"),
            }
            if trig not in action_map:
                return dash.no_update
            src, dst = action_map[trig]
            with self._lock:
                cells = []
                if cell_sel == "ALL" or not cell_sel:
                    cells = list(self.cell_list.values())
                else:
                    c = self.cell_list.get(cell_sel)
                    if c:
                        cells = [c]
                applied = 0
                for c in cells:
                    if hasattr(c, "adjust_slice_quota_move_rb"):
                        moved = c.adjust_slice_quota_move_rb(src, dst, prb_step=SLICE_MOVE_STEP_PRBS)
                        applied += 1 if moved else 0
                if applied == 0:
                    return f"Move {src}→{dst}: no change (insufficient quota or invalid cell)."
                # Preview quotas on first affected cell
                any_cell = cells[0] if cells else None
                if any_cell:
                    preview = any_cell.slice_dl_prb_quota
                    return f"Moved {SLICE_MOVE_STEP_PRBS} PRBs from {src} to {dst} on {cell_sel if cell_sel!='ALL' else 'ALL cells'}. Example {any_cell.cell_id} quotas: {preview} (max {any_cell.max_dl_prb})."
                return f"Moved {SLICE_MOVE_STEP_PRBS} PRBs from {src} to {dst}."

        @app.callback(
            Output("ue-bitrate", "figure"),
            Output("ue-sinr", "figure"),
            Output("ue-cqi", "figure"),
            Output("ue-prb-granted", "figure"),
            Output("ue-prb-requested", "figure"),
            Output("cell-load", "figure"),
            Output("ue-buffer", "figure"),
            Output("ue-slice-list", "children"),
            Output("cell-quota-live", "children"),
            Output("ue-dl-latency", "figure"),
            Output("ue-list", "options"),
            Input("tick", "n_intervals"),
        )
        def _update(_n):
            with self._lock:
                tx = list(self._t)
                if not tx:
                    empty = go.Figure()
                    return empty, empty, empty, empty, empty, empty, empty, "", "", empty, []

                ue_keys = list(set(
                    list(self._ue_dl_mbps.keys())
                    + list(self._ue_sinr_db.keys())
                    + list(self._ue_cqi.keys())
                    + list(self._ue_dl_buf.keys())
                    + list(self._ue_dl_prb.keys())
                    + list(getattr(self, "_ue_dl_prb_req", {}).keys())
                ))
                # Apply UE filter if set
                if getattr(self, "_ue_filter", set()):
                    ue_keys = [k for k in ue_keys if k in self._ue_filter]

                cell_keys = list(set(
                    list(self._cell_dl_load.keys())
                    + list(self._cell_alloc_prb.keys())
                    + list(self._cell_max_prb.keys())
                ))

                # Build time axis in seconds (trace replay clock): step * dt * speedup
                dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0) or 1.0)
                speed = float(getattr(settings, "TRACE_SPEEDUP", 1.0) or 1.0)
                tx_s = [t * dt * speed for t in tx]

                # Label helper and color selection by slice
                slice_map = {}
                try:
                    for imsi in ue_keys:
                        ue_obj = self.ue_list.get(imsi)
                        slice_map[imsi] = getattr(ue_obj, "slice_type", None)
                except Exception:
                    pass
                def _label(imsi, suffix):
                    sl = slice_map.get(imsi)
                    sl_str = sl if sl is not None else "?"
                    return f"{imsi} ({sl_str}) {suffix}"

                # --- UE bitrate (Mbps) ---
                tr_bitrate = []
                for imsi in ue_keys:
                    ys = list(self._ue_dl_mbps.get(imsi, []))
                    if ys:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_bitrate.append(go.Scatter(x=tx_s[-len(ys):], y=ys, mode="lines", name=_label(imsi, "DL Mbps"), line=dict(color=c)))

                # --- UE SINR ---
                tr_sinr = []
                for imsi in ue_keys:
                    ys_s = list(self._ue_sinr_db.get(imsi, []))
                    if ys_s:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_sinr.append(go.Scatter(x=tx_s[-len(ys_s):], y=ys_s, mode="lines", name=_label(imsi, "SINR (dB)"), line=dict(color=c)))

                # --- UE CQI ---
                tr_cqi = []
                for imsi in ue_keys:
                    ys_c = list(self._ue_cqi.get(imsi, []))
                    if ys_c:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_cqi.append(go.Scatter(x=tx_s[-len(ys_c):], y=ys_c, mode="lines", name=_label(imsi, "CQI"), line=dict(color=c)))

                # --- UE DL PRBs: GRANTED ---
                tr_prb_granted = []
                for imsi in ue_keys:
                    ys_g = list(self._ue_dl_prb.get(imsi, []))
                    if ys_g:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_prb_granted.append(go.Scatter(x=tx_s[-len(ys_g):], y=ys_g, mode="lines", name=_label(imsi, "granted"), line=dict(color=c)))

                # --- UE DL PRBs: REQUESTED ---
                tr_prb_requested = []
                for imsi in ue_keys:
                    ys_r = list(getattr(self, "_ue_dl_prb_req", {}).get(imsi, []))
                    if ys_r:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_prb_requested.append(go.Scatter(x=tx_s[-len(ys_r):], y=ys_r, mode="lines", name=_label(imsi, "requested"), line=dict(color=c)))



                tr_latency = []
                for imsi in ue_keys:
                    ys = list(self._ue_dl_latency.get(imsi, []))
                    if ys:
                        # Convert seconds to milliseconds for plotting
                        ys_ms = [v * 1000.0 for v in ys]
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_latency.append(go.Scatter(x=tx_s[-len(ys_ms):], y=ys_ms, mode="lines", name=_label(imsi, "DL latency (ms)"), line=dict(color=c)))
                
               
                # --- Cell load & PRBs ---
                tr_cell = []
                for cid in cell_keys:
                    ys = list(self._cell_dl_load.get(cid, []))
                    if ys:
                        tr_cell.append(go.Scatter(x=tx_s[-len(ys):], y=ys, mode="lines", name=f"{cid} DL load"))
                for cid in cell_keys:
                    ys_a = list(self._cell_alloc_prb.get(cid, []))
                    if ys_a:
                        tr_cell.append(go.Scatter(x=tx_s[-len(ys_a):], y=ys_a, mode="lines", name=f"{cid} alloc PRB", line={"dash": "dot"}))
                    ys_m = list(self._cell_max_prb.get(cid, []))
                    if ys_m:
                        tr_cell.append(go.Scatter(x=tx_s[-len(ys_m):], y=ys_m, mode="lines", name=f"{cid} max PRB", line={"dash": "dash"}))

                # --- UE DL buffer (optional) ---
                tr_buf = []
                for imsi in ue_keys:
                    ys = list(self._ue_dl_buf.get(imsi, []))
                    if ys:
                        c = SLICE_COLORS.get(slice_map.get(imsi))
                        tr_buf.append(go.Scatter(x=tx_s[-len(ys):], y=ys, mode="lines", name=_label(imsi, "DL buffer (bytes)"), line=dict(color=c)))

            fig_bitrate = tidy(go.Figure(data=tr_bitrate), "Per‑UE Downlink Bitrate (Mbps)", "Mbps", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_sinr = tidy(go.Figure(data=tr_sinr), "Per‑UE SINR", "SINR (dB)", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_cqi = tidy(go.Figure(data=tr_cqi), "Per‑UE CQI", "CQI", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_prb_granted = tidy(go.Figure(data=tr_prb_granted), "Per‑UE DL PRBs — GRANTED", "PRBs", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_prb_requested = tidy(go.Figure(data=tr_prb_requested), "Per‑UE DL PRBs — REQUESTED", "PRBs", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_cell = tidy(go.Figure(data=tr_cell), "Per‑Cell Load & PRBs", "Value / PRBs", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_buf = tidy(go.Figure(data=tr_buf), "Per‑UE DL Buffer (bytes)*", "Bytes", show_legend=self._show_legend, overlay_legend=self._overlay_legend)
            fig_latency = tidy(go.Figure(data=tr_latency), "Per‑UE Downlink Latency", "Milliseconds", show_legend=self._show_legend, overlay_legend=self._overlay_legend)

            # Live per-cell quotas and last RL action (if present)
            quota_lines = []
            for cid, cell in self.cell_list.items():
                q = getattr(cell, "slice_dl_prb_quota", {}) or {}
                e = int(q.get("eMBB", 0) or 0)
                u = int(q.get("URLLC", 0) or 0)
                m = int(q.get("mMTC", 0) or 0)
                maxp = int(getattr(cell, "max_dl_prb", 0) or 0)
                act = getattr(cell, "rl_last_action", None)
                if isinstance(act, dict):
                    a_lbl = act.get("label", "keep")
                    a_mv = act.get("moved", 0)
                    a_step = act.get("step", "-")
                    who = act.get("actor", "?")
                    src_engine = act.get("source")
                    engine_tag = f" ({src_engine})" if src_engine else ""
                    extra = f" | last {who}{engine_tag}: {a_lbl} (+{a_mv}) at step {a_step}"
                else:
                    extra = ""
                quota_lines.append(f"{cid}: eMBB={e}, URLLC={u}, mMTC={m} (max {maxp}){extra}")
            quota_text = "\n".join(quota_lines)
            # UE→slice preview string (compact)
            ue_preview = ", ".join([f"{imsi}→{slice_map.get(imsi) or '?'}" for imsi in sorted(ue_keys)])
            ue_slice_children = html.Div([html.Strong("UE→Slice: "), html.Span(ue_preview)])

            # Build UE options for the external legend from ALL known UEs (not only those with recent samples)
            all_imsi = sorted(self.ue_list.keys())
            ue_options = [
                {"label": f"{imsi} ({getattr(self.ue_list.get(imsi), 'slice_type', None) or '?'} )", "value": imsi}
                for imsi in all_imsi
            ]

            return fig_bitrate, fig_sinr, fig_cqi, fig_prb_granted, fig_prb_requested, fig_cell, fig_buf, ue_slice_children, quota_text, fig_latency, ue_options

        @app.callback(
            Output("ue-list", "value"),
            Input("ue-list", "options"),
            Input("btn-ue-select-all", "n_clicks"),
            Input("btn-ue-clear", "n_clicks"),
            State("ue-list", "value"),
        )
        def _sync_ue_list(options, n_all, n_clear, cur):
            # Determine which trigger fired
            ctx = dash.callback_context
            trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx and ctx.triggered else None
            all_vals = [opt["value"] for opt in (options or [])]
            if trig == "btn-ue-select-all":
                return all_vals
            if trig == "btn-ue-clear":
                return []
            # Initialize selection to ALL when options change and current empty/None
            if (not cur) and options:
                return all_vals
            return dash.no_update

        def _run():
            app.run_server(host="127.0.0.1", port=DASH_PORT, debug=False)

        self._dash_thread = threading.Thread(target=_run, daemon=True)
        self._dash_thread.start()
        print(f"{self.xapp_id}: live KPI dashboard at http://localhost:{DASH_PORT}")

    def __del__(self):
        # Best-effort close of log file handles
        try:
            if self._ue_log_fp:
                self._ue_log_fp.close()
        except Exception:
            pass
        try:
            if self._cell_log_fp:
                self._cell_log_fp.close()
        except Exception:
            pass
