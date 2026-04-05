"""
visualization.py
────────────────
All Cytoscape / Dash visualization logic.

Design
------
- One permanent "⚙ Configuration" tab, never rebuilt.
- Every time a simulation is launched, a new "▶ Sim N" tab is appended.
- Each sim tab is fully independent: own interval, own stores, own graph.
- Each sim has a Pause / Resume button backed by a threading.Event.
- The config tab is shown/hidden via CSS so its form state is never lost.
"""

import os
import sys
import threading

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import plotly.graph_objects as go
from dash import Input, Output, State, ALL, MATCH, ctx, dcc, html

from flask_server import get_server


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "background-color": "data(color)",
            "label": "data(label)",
            "border-width": 2,
            "border-color": "black",
            "border-style": "solid",
        },
    },
    {"selector": "edge", "style": {"line-color": "#ccc"}},
]

CONFIG_TAB_VALUE = "config-tab"

# Colour palette for per-class lines (consistent across all plots)
_CLASS_COLORS = {
    "MoodySARSA": "#9b59b6",
    "SARSA":      "#f39c12",
    "TFT":        "#27ae60",
    "WSLS":       "#2980b9",
}

_PLOT_LAYOUT = dict(
    margin=dict(l=28, r=8, t=8, b=28),
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font=dict(size=9),
    legend=dict(font=dict(size=8), orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="#e9ecef", title="Round"),
    yaxis=dict(showgrid=True, gridcolor="#e9ecef"),
)


def _empty_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_PLOT_LAYOUT,
        annotations=[dict(text="No data yet", x=0.5, y=0.5,
                          xref="paper", yref="paper",
                          showarrow=False, font=dict(color="#adb5bd", size=10))],
    )
    return fig


def _build_line_figure(series: dict, metric: str, y_title: str = "",
                        y_range: list | None = None) -> go.Figure:
    """Build a multi-line Plotly figure from a metrics series dict."""
    rounds = series.get("rounds", [])
    classes = series.get(metric, {})
    if not rounds or not any(v for v in classes.values()):
        return _empty_figure()

    fig = go.Figure()
    for cls, values in classes.items():
        if not values:
            continue
        fig.add_trace(go.Scatter(
            x=rounds, y=values,
            mode="lines",
            name=cls,
            line=dict(color=_CLASS_COLORS.get(cls, "#999"), width=1.5),
        ))
    layout = dict(**_PLOT_LAYOUT)
    if y_range:
        layout["yaxis"] = {**_PLOT_LAYOUT["yaxis"], "range": y_range}
    if y_title:
        layout["yaxis"] = {**layout.get("yaxis", {}), "title": y_title}
    fig.update_layout(**layout)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CytoscapeRenderer
# ──────────────────────────────────────────────────────────────────────────────

class CytoscapeRenderer:
    """Converts NetworkX graphs into Dash Cytoscape element lists."""

    @staticmethod
    def elements(
        graph: nx.Graph,
        colors: list | dict,
        dimensions: int,
        moods: list | dict | None = None,
        positions: dict | None = None,
    ) -> list[dict]:
        offset = 12.5
        result = []

        for node in graph.nodes():
            color = (
                colors[node] if isinstance(colors, dict)
                else (colors[node] if node < len(colors) else "orange")
            )
            mood = (
                moods[node] if moods and isinstance(moods, dict)
                else (moods[node] if moods and node < len(moods) else None)
            )
            if positions and node in positions:
                position = positions[node]
            else:
                row, col = divmod(node, dimensions)
                position = {
                    "x": col * 125 + (row % 2) * offset,
                    "y": row * 125 + (col % 2) * offset,
                }
            result.append({
                "data": {
                    "id": str(node),
                    "label": f"{node} | Mood: {mood}" if mood is not None else f"Node {node}",
                    "color": color,
                    "mood": mood,
                },
                "position": position,
                "classes": color,
            })

        for u, v in graph.edges():
            result.append({"data": {"source": str(u), "target": str(v)}})

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-simulation state
# ──────────────────────────────────────────────────────────────────────────────

class SimulationState:
    """
    Holds live display data and pause control for one simulation run.
    Thread-safe for single-writer / single-reader use.
    """

    def __init__(self, sim_id: str, graph: nx.Graph, dimensions: int,
                 positions: dict, initial_colors: list, sim_name: str = "",
                 config: dict | None = None):
        self.sim_id = sim_id
        self.sim_name = sim_name
        self.graph = graph
        self.dimensions = dimensions
        self.positions = positions
        self.colors: list = list(initial_colors)
        self.moods: list = []
        self.paused: bool = False
        self.finished: bool = False
        self.config: dict = config or {}
        self.iteration: int = 0  # updated by sim thread each reconstruction

        # threading.Event: set = running, clear = paused
        self._resume_event = threading.Event()
        self._resume_event.set()  # start in running state
        # Lock protecting colors/moods against concurrent read/write
        self._data_lock = threading.Lock()
        # Latest metrics series snapshot — written by sim thread, read by Dash
        self._metrics: dict = {}
        self._metrics_lock = threading.Lock()

    # ── Pause control (called from Dash callbacks) ─────────────────────────

    def pause(self):
        self.paused = True
        self._resume_event.clear()

    def resume(self):
        self.paused = False
        self._resume_event.set()

    def toggle_pause(self):
        if self.paused:
            self.resume()
        else:
            self.pause()

    def wait_if_paused(self):
        """Called by the simulation thread on each iteration."""
        self._resume_event.wait()

    # ── Data update (called from simulation thread) ────────────────────────

    def update(self, colors: list, moods: list):
        """
        Write new display data atomically.
        The Dash interval callback reads colors/moods from the main thread
        while the sim thread writes here — the lock prevents torn reads.
        """
        with self._data_lock:
            self.colors = colors
            self.moods  = moods

    def read(self) -> tuple[list, list]:
        """Read colors and moods atomically."""
        with self._data_lock:
            return list(self.colors), list(self.moods)

    def update_iteration(self, i: int):
        """Called from the sim thread each reconstruction event."""
        self.iteration = i  # int write is atomic in CPython — no lock needed

    def update_metrics(self, series: dict):
        """Called from the sim thread after each tracker snapshot."""
        with self._metrics_lock:
            self._metrics = series

    def read_metrics(self) -> dict:
        """Called from the Dash interval callback."""
        with self._metrics_lock:
            return dict(self._metrics)


# ──────────────────────────────────────────────────────────────────────────────
# Sim tab layout builder
# ──────────────────────────────────────────────────────────────────────────────

class SimTabLayout:
    """Builds the layout for one simulation tab. All IDs are namespaced by sim_id."""

    def __init__(self, state: SimulationState):
        self.state = state

    # ── Style constants ───────────────────────────────────────────────────
    _PARAM_LABEL = {"fontSize": "0.7rem", "color": "#868e96",
                    "textTransform": "uppercase", "letterSpacing": "0.04em",
                    "marginBottom": "1px", "fontWeight": "600"}
    _PARAM_VALUE = {"fontSize": "0.82rem", "color": "#212529",
                    "marginBottom": "8px", "fontWeight": "400"}
    _TOOLBAR_H   = "46px"
    _LEFT_W      = "220px"

    def _param_row(self, label: str, value) -> html.Div:
        return html.Div([
            html.Div(label, style=self._PARAM_LABEL),
            html.Div(str(value), style=self._PARAM_VALUE),
        ])

    def _plot_card(self, title: str, graph_id: dict) -> html.Div:
        return html.Div([
            html.Div(title, style={"fontSize": "0.72rem", "fontWeight": "600",
                                   "color": "#495057", "marginBottom": "1px"}),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False},
                style={"height": "145px"},
                figure=_empty_figure(),
            ),
        ], style={"marginBottom": "6px"})

    def build(self) -> html.Div:
        sid  = self.state.sim_id
        name = self.state.sim_name if self.state.sim_name else f"Sim {sid}"
        cfg  = self.state.config
        initial_elements = CytoscapeRenderer.elements(
            self.state.graph, self.state.colors,
            self.state.dimensions, positions=self.state.positions,
        )

        # ── Left panel: params + iteration counter ────────────────────────
        def _fmt_dist(v):
            return "∞" if float(v) > 100_000 else str(v)

        agent_counts = {}
        from collections import Counter
        for color in self.state.colors:
            from simulation_core import COLOR_AGENT_MAP
            agent_counts[COLOR_AGENT_MAP.get(color, color)] =                 agent_counts.get(COLOR_AGENT_MAP.get(color, color), 0) + 1

        agent_rows = [
            self._param_row(atype, f"{count} agents")
            for atype, count in sorted(agent_counts.items())
            if count > 0
        ]

        left_panel = html.Div([
            # ── Iteration counter (updated by callback) ───────────────
            html.Div([
                html.Div("Iteration", style=self._PARAM_LABEL),
                html.Div(
                    "0",
                    id={"type": "iteration-display", "sim": sid},
                    style={**self._PARAM_VALUE,
                           "fontSize": "1.4rem", "fontWeight": "700",
                           "color": "#2c3e50", "marginBottom": "10px"},
                ),
            ]),
            html.Hr(style={"margin": "6px 0 10px"}),

            # ── Static config params ──────────────────────────────────
            html.Div("Configuration", style={**self._PARAM_LABEL,
                                             "marginBottom": "6px"}),
            self._param_row("Nodes",        cfg.get("num_nodes", "—")),
            self._param_row("Edges",        cfg.get("num_edges", "—")),
            self._param_row("Total games",  cfg.get("num_games_per_pair", "—")),
            self._param_row("Reconnect %",
                f'{cfg.get("percent_reconnection", 0) * 100:.0f}%'),
            self._param_row("Max distance",
                _fmt_dist(cfg.get("max_connection_distance", 0))),
            self._param_row("Betrayal thr.",
                cfg.get("average_considered_betrayal", "—")),
            self._param_row("Edge mode",    cfg.get("forgiveness_mode", "—")),

            html.Hr(style={"margin": "6px 0 10px"}),
            html.Div("Agent Composition", style={**self._PARAM_LABEL,
                                                  "marginBottom": "6px"}),
            *agent_rows,

            html.Hr(style={"margin": "6px 0 10px"}),

            # ── Analysis plots (burger menu replaced by inline panel) ─
            html.Div("Live Analysis", style={**self._PARAM_LABEL,
                                             "marginBottom": "4px"}),
            html.Div(
                "Waiting for data…",
                id={"type": "sidebar-waiting", "sim": sid},
                style={"fontSize": "0.72rem", "color": "#adb5bd",
                       "fontStyle": "italic", "marginBottom": "4px"},
            ),
            self._plot_card("Avg Score / Round",
                            {"type": "plot-score",        "sim": sid}),
            self._plot_card("Avg Connections",
                            {"type": "plot-connectivity",  "sim": sid}),
            self._plot_card("Cooperation Rate",
                            {"type": "plot-cooperation",   "sim": sid}),
            self._plot_card("MoodySARSA Mood",
                            {"type": "plot-mood",          "sim": sid}),

            # Stores live here so they exist regardless of sidebar state
            dcc.Store(id={"type": "metrics-store", "sim": sid}, data={}),
            dcc.Store(id={"type": "color-store",   "sim": sid},
                      data=self.state.colors),
            dcc.Store(id={"type": "mood-store",    "sim": sid}, data=[]),
            dcc.Interval(id={"type": "sim-interval", "sim": sid},
                         interval=500, n_intervals=0),

        ], style={
            "width": self._LEFT_W,
            "minWidth": self._LEFT_W,
            "height": f"calc(100vh - {self._TOOLBAR_H})",
            "overflowY": "auto",
            "padding": "12px 10px",
            "borderRight": "1px solid #dee2e6",
            "backgroundColor": "#f8f9fa",
            "flexShrink": "0",
        })

        # ── Graph panel ───────────────────────────────────────────────────
        graph_panel = html.Div([
            cyto.Cytoscape(
                id={"type": "cytoscape", "sim": sid},
                layout={"name": "preset"},
                elements=initial_elements,
                zoom=1,
                pan={"x": 0, "y": 0},
                style={"width": "100%",
                       "height": f"calc(100vh - {self._TOOLBAR_H})"},
                stylesheet=CYTO_STYLESHEET,
            ),
        ], style={"flex": "1", "minWidth": "0", "position": "relative"})

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = html.Div([
            html.Strong(name, style={"fontSize": "0.95rem",
                                     "marginRight": "auto"}),
            html.Button(
                "⏸ Pause",
                id={"type": "pause-btn", "sim": sid},
                n_clicks=0,
                style={"backgroundColor": "#f39c12", "color": "white",
                       "padding": "4px 14px", "border": "none",
                       "borderRadius": "4px", "cursor": "pointer",
                       "marginRight": "8px", "fontSize": "0.82rem"},
            ),
            html.Button(
                "⟳ Restart",
                id={"type": "restart-btn", "sim": sid},
                n_clicks=0,
                style={"backgroundColor": "#e74c3c", "color": "white",
                       "padding": "4px 14px", "border": "none",
                       "borderRadius": "4px", "cursor": "pointer",
                       "fontSize": "0.82rem"},
            ),
            html.Span(id={"type": "pause-status", "sim": sid},
                      className="ms-3 text-muted",
                      style={"fontSize": "0.78rem"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "height": self._TOOLBAR_H,
            "padding": "0 14px",
            "borderBottom": "1px solid #dee2e6",
            "backgroundColor": "#ffffff",
            "flexShrink": "0",
        })

        return html.Div([
            toolbar,
            html.Div([left_panel, graph_panel],
                     style={"display": "flex", "flex": "1",
                            "minHeight": "0", "overflow": "hidden"}),
        ], style={
            "display": "flex", "flexDirection": "column",
            "height": "100vh", "overflow": "hidden",
        })


# ──────────────────────────────────────────────────────────────────────────────
# SimulationApp
# ──────────────────────────────────────────────────────────────────────────────

class SimulationApp:
    """
    Single Dash app hosting:
      • One permanent config tab (never rebuilt, never resets form inputs).
      • One new tab per simulation launched, each fully independent.

    Public API
    ----------
    attach_simulation(graph, dims, positions, colors) -> SimulationState
        Call this when launching a sim. Returns the SimulationState the sim
        thread should call .update() and .wait_if_paused() on.

    app.run(debug=True)
    """

    def __init__(self, server, config_layout: html.Div):
        self._config_layout = config_layout
        self._simulations: list[SimulationState] = []
        self._sim_counter = 0
        self._lock = threading.Lock()

        self.app = dash.Dash(
            __name__,
            server=server,
            url_base_pathname="/",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.app.layout = self._build_shell()
        self._register_callbacks()

    # ── Public API ────────────────────────────────────────────────────────────

    def attach_simulation(
        self,
        graph: nx.Graph,
        dimensions: int,
        positions: dict,
        initial_colors: list,
        sim_name: str = "",
        config: dict | None = None,
    ) -> SimulationState:
        """
        Register a new simulation. Returns SimulationState for the caller
        to pass to the sim thread.
        """
        with self._lock:
            self._sim_counter += 1
            sid = str(self._sim_counter)
            state = SimulationState(sid, graph, dimensions, positions,
                                    initial_colors, sim_name=sim_name,
                                    config=config)
            self._simulations.append(state)
        self._register_sim_visibility(sid)
        return state

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_shell(self) -> dbc.Container:
        """
        Static shell. Sim divs are appended dynamically; only their outer
        wrapper style changes — inner content is never touched after creation.
        """
        return dbc.Container([
            dcc.Tabs(id="main-tabs", value=CONFIG_TAB_VALUE, children=[
                dcc.Tab(label="⚙ Configuration", value=CONFIG_TAB_VALUE),
            ]),

            # Config content — always in DOM, toggled via clientside callback
            html.Div(self._config_layout, id="config-tab-content",
                     style={"display": "block"}),

            # Sim divs appended here — never rebuilt after creation
            html.Div(id="sim-tabs-container"),

            dcc.Interval(id="shell-interval", interval=1000, n_intervals=0),
            dcc.Store(id="known-sim-count", data=0),
            # Stores the currently active tab so clientside CB can read it
            dcc.Store(id="active-tab-store", data=CONFIG_TAB_VALUE),
        ], fluid=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _register_callbacks(self):
        self._cb_detect_new_sims()
        self._cb_show_hide_tabs()
        self._cb_sync_stores()
        self._cb_update_graphs()
        self._cb_update_iteration()
        self._cb_pause_buttons()
        self._cb_restart_buttons()
        self._cb_sync_metrics()
        self._cb_update_plots()

    def _cb_detect_new_sims(self):
        """
        Polls once per second. Only appends NEW sim divs — never touches
        existing ones so button state and graph zoom are never reset.
        Also updates active-tab-store so the clientside visibility CB fires.
        """
        @self.app.callback(
            Output("main-tabs",          "children"),
            Output("sim-tabs-container", "children"),
            Output("known-sim-count",    "data"),
            Output("main-tabs",          "value"),
            Input("shell-interval",      "n_intervals"),
            State("sim-tabs-container",  "children"),
            State("known-sim-count",     "data"),
            State("main-tabs",           "value"),
        )
        def detect_new_sims(_, sim_divs, known_count, active_tab):
            with self._lock:
                total = len(self._simulations)
                all_states = list(self._simulations)

            # Tab bar — always correct order, cheap to rebuild (labels only)
            tab_children = [dcc.Tab(label="⚙ Configuration", value=CONFIG_TAB_VALUE)]
            for state in all_states:
                display_name = state.sim_name if state.sim_name else f"Sim {state.sim_id}"
                indicator = " ✓" if state.finished else ""
                tab_children.append(
                    dcc.Tab(
                        label=f"▶ {display_name}{indicator}",
                        value=f"sim-{state.sim_id}",
                    )
                )

            if total <= known_count:
                return tab_children, dash.no_update, known_count, active_tab

            # Append divs for NEW sims only — existing divs are untouched
            existing = list(sim_divs) if sim_divs else []
            for state in all_states[known_count:]:
                sid = state.sim_id
                existing.append(
                    html.Div(
                        SimTabLayout(state).build(),
                        id=f"sim-div-{sid}",
                        style={"display": "none"},
                    )
                )

            # Switch to newest sim — clientside callback sees the tab change
            # and immediately shows the correct div in the browser.
            new_active = f"sim-{all_states[-1].sim_id}"
            return tab_children, existing, total, new_active

    def _cb_show_hide_tabs(self):
        """
        Clientside callback: reads active-tab-store and toggles display on
        every sim wrapper div and the config div — pure JS, zero server
        round-trip, fires after the DOM has been updated with new divs.
        """
        @self.app.callback(
            Output("config-tab-content", "style"),
            Input("main-tabs", "value"),
        )
        def toggle_config(active_tab):
            return {"display": "block" if active_tab == CONFIG_TAB_VALUE else "none"}

        # Clientside JS: fires on every tab change (user click or new sim).
        # Reads main-tabs directly so there is exactly one output owner for
        # active-tab-store. Shows the matching sim-div-*, hides all others.
        self.app.clientside_callback(
            """
            function(activeTab) {
                var simDivs = document.querySelectorAll('[id^="sim-div-"]');
                simDivs.forEach(function(div) {
                    var sid = div.id.replace('sim-div-', '');
                    div.style.display = (activeTab === 'sim-' + sid) ? 'block' : 'none';
                });
                return activeTab;
            }
            """,
            Output("active-tab-store", "data"),
            Input("main-tabs", "value"),
        )

    def _register_sim_visibility(self, sid: str):
        # No longer needed — clientside callback handles all sim divs.
        pass

    def _cb_sync_stores(self):
        """
        Each sim's interval pumps its state into its own stores.
        Returns no_update when paused so the graph stops refreshing
        and the pause button label is never reset.
        """
        @self.app.callback(
            Output({"type": "color-store", "sim": MATCH}, "data"),
            Output({"type": "mood-store",  "sim": MATCH}, "data"),
            Input({"type": "sim-interval", "sim": MATCH}, "n_intervals"),
            State({"type": "sim-interval", "sim": MATCH}, "id"),
        )
        def sync(_, interval_id):
            sid = interval_id["sim"]
            state = self._find_state(sid)
            if state is None or state.paused:
                return dash.no_update, dash.no_update
            # Use atomic read so we never return a half-written list
            colors, moods = state.read()
            return colors, moods

    def _cb_update_graphs(self):
        """Each sim's stores trigger a re-render of its own Cytoscape graph."""
        @self.app.callback(
            Output({"type": "cytoscape", "sim": MATCH}, "elements"),
            Output({"type": "cytoscape", "sim": MATCH}, "zoom"),
            Output({"type": "cytoscape", "sim": MATCH}, "pan"),
            Input({"type": "color-store", "sim": MATCH}, "data"),
            Input({"type": "mood-store",  "sim": MATCH}, "data"),
            State({"type": "cytoscape",   "sim": MATCH}, "zoom"),
            State({"type": "cytoscape",   "sim": MATCH}, "pan"),
            State({"type": "color-store", "sim": MATCH}, "id"),
        )
        def update_graph(colors, moods, zoom, pan, store_id):
            if not colors:
                return dash.no_update, zoom, pan
            sid = store_id["sim"]
            state = self._find_state(sid)
            if state is None:
                return dash.no_update, zoom, pan
            elements = CytoscapeRenderer.elements(
                state.graph, colors, state.dimensions,
                moods=moods, positions=state.positions,
            )
            return elements, zoom, pan

    def _cb_update_iteration(self):
        """Update the iteration counter display on each interval tick."""
        @self.app.callback(
            Output({"type": "iteration-display", "sim": MATCH}, "children"),
            Input({"type": "sim-interval", "sim": MATCH}, "n_intervals"),
            State({"type": "sim-interval", "sim": MATCH}, "id"),
        )
        def update_iter(_, interval_id):
            sid   = interval_id["sim"]
            state = self._find_state(sid)
            if state is None:
                return dash.no_update
            return f"{state.iteration:,}"

    def _cb_pause_buttons(self):
        """Pause/Resume button for each sim tab."""
        @self.app.callback(
            Output({"type": "pause-status", "sim": MATCH}, "children"),
            Output({"type": "pause-btn",    "sim": MATCH}, "children"),
            Output({"type": "pause-btn",    "sim": MATCH}, "style"),
            Input({"type": "pause-btn",     "sim": MATCH}, "n_clicks"),
            State({"type": "pause-btn",     "sim": MATCH}, "id"),
            prevent_initial_call=True,
        )
        def toggle_pause(n_clicks, btn_id):
            sid = btn_id["sim"]
            state = self._find_state(sid)
            if state is None:
                return dash.no_update, dash.no_update, dash.no_update
            state.toggle_pause()
            if state.paused:
                return (
                    "⏸ Simulation paused",
                    "▶ Resume",
                    {
                        "backgroundColor": "#27ae60", "color": "white",
                        "padding": "8px 18px", "border": "none",
                        "borderRadius": "4px", "cursor": "pointer",
                        "marginRight": "10px",
                    },
                )
            return (
                "▶ Simulation running",
                "⏸ Pause",
                {
                    "backgroundColor": "#f39c12", "color": "white",
                    "padding": "8px 18px", "border": "none",
                    "borderRadius": "4px", "cursor": "pointer",
                    "marginRight": "10px",
                },
            )

    def _cb_restart_buttons(self):
        """Restart button restarts the whole server process."""
        @self.app.callback(
            Output({"type": "pause-status", "sim": MATCH}, "children",
                   allow_duplicate=True),
            Input({"type": "restart-btn", "sim": MATCH}, "n_clicks"),
            prevent_initial_call=True,
        )
        def restart(n_clicks):
            if n_clicks:
                print("Restarting server...")
                os.execl(sys.executable, sys.executable, *sys.argv)
            return dash.no_update

    def _cb_sync_metrics(self):
        """
        Pump the latest metrics series from SimulationState into the per-sim
        metrics-store on the same 500ms interval used for color/mood.
        Returns no_update when paused so plots freeze cleanly.
        """
        @self.app.callback(
            Output({"type": "metrics-store", "sim": MATCH}, "data"),
            Input({"type": "sim-interval",   "sim": MATCH}, "n_intervals"),
            State({"type": "sim-interval",   "sim": MATCH}, "id"),
        )
        def sync_metrics(_, interval_id):
            sid   = interval_id["sim"]
            state = self._find_state(sid)
            if state is None or state.paused:
                return dash.no_update
            metrics = state.read_metrics()
            if not metrics:
                return dash.no_update
            return metrics

    def _cb_update_plots(self):
        """
        Rebuild all four sidebar plots whenever the metrics-store updates.
        One callback handles all four figures to avoid four separate server
        round-trips per interval tick.
        """
        @self.app.callback(
            Output({"type": "plot-score",       "sim": MATCH}, "figure"),
            Output({"type": "plot-connectivity","sim": MATCH}, "figure"),
            Output({"type": "plot-cooperation", "sim": MATCH}, "figure"),
            Output({"type": "plot-mood",        "sim": MATCH}, "figure"),
            Output({"type": "sidebar-waiting",  "sim": MATCH}, "style"),
            Input({"type": "metrics-store",     "sim": MATCH}, "data"),
        )
        def update_plots(metrics):
            if not metrics or not metrics.get("rounds"):
                no_data_style = {"fontSize": "0.78rem", "color": "#868e96",
                                 "fontStyle": "italic", "marginBottom": "8px"}
                return (
                    _empty_figure(), _empty_figure(),
                    _empty_figure(), _empty_figure(),
                    no_data_style,
                )

            hidden = {"display": "none"}
            return (
                _build_line_figure(metrics, "score",
                                   y_title="Avg payoff"),
                _build_line_figure(metrics, "connectivity",
                                   y_title="Norm. degree", y_range=[0, 1]),
                _build_line_figure(metrics, "cooperation",
                                   y_title="Coop. rate",   y_range=[0, 1]),
                _build_line_figure(metrics, "mood",
                                   y_title="Mood (0–100)", y_range=[0, 100]),
                hidden,
            )

    # _cb_sidebar removed — analysis panel is now a permanent left panel
    # in the two-column layout, no toggle needed.

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_state(self, sid: str) -> SimulationState | None:
        with self._lock:
            for s in self._simulations:
                if s.sim_id == sid:
                    return s
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Legacy shims
# ──────────────────────────────────────────────────────────────────────────────

def nx_to_cytoscape(graph, colors, dimensions, moods=None, positions=None):
    return CytoscapeRenderer.elements(graph, colors, dimensions, moods, positions)


def cytoscape_with_layout(graph, colors, dimensions, positions):
    elements = CytoscapeRenderer.elements(graph, colors, dimensions, positions=positions)
    return cyto.Cytoscape(
        id="cytoscape-graph-legacy",
        elements=elements,
        layout={"name": "preset"},
        zoom=1,
        pan={"x": 0, "y": 0},
        style={"width": "100%", "height": "700px"},
        stylesheet=CYTO_STYLESHEET,
    )