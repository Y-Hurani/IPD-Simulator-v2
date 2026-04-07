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
from dash import ClientsideFunction, Input, Output, State, ALL, MATCH, ctx, dcc, html

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
        # Version counters — incremented on each write so Dash can skip
        # serialisation when nothing has changed since the last poll.
        # CPython int assignment is atomic under the GIL — no lock needed.
        self._graph_version: int = 0
        self._metrics_version: int = 0
        # Backpressure: _acked_version tracks the last version the Dash
        # callback has successfully consumed. try_update() only bumps
        # _graph_version when _acked_version has caught up, preventing the
        # sim thread from queuing more updates than the server can send.
        self._acked_version: int = 0
        # Delta tracking — computed each update(), read by Dash callback
        self._prev_colors: list | None = None    # colors from last update()
        self._color_delta: dict = {}             # {str(node_id): color}
        self._prev_edges: set | None = None      # frozenset of (a,b) tuples
        self._edge_delta: dict = {"added": [], "removed": []}
        # Set to False by update_visibility when the tab is hidden.
        # update() skips all delta computation (including the expensive edge
        # frozenset) when False — sim thread still runs, just doesn't diff.
        # When tab becomes visible, reset to True and _prev_edges = None so
        # the next send triggers a clean full repaint with no stale deltas.
        self.rendering_active: bool = False

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
        Write latest colors/moods and compute deltas. Always updates the
        raw data so the sim stays current, but only signals the Dash
        callback (by incrementing _graph_version) when the previous frame
        has been acknowledged. This is the backpressure gate.

        If the browser hasn't consumed the last frame yet:
          - colors/moods are refreshed in place (sim stays current)
          - deltas accumulate (the next send will cover all missed changes)
          - _graph_version is NOT bumped → Dash sees no new version → no send

        If the browser has consumed the last frame (_acked_version caught up):
          - accumulated deltas are reset (browser now has a fresh baseline)
          - new deltas are computed from the current state
          - _graph_version is bumped → Dash sends

        Delta reset happens HERE, not in acknowledge(). acknowledge() only
        records the ack version. This ensures that if the browser is slow or
        backgrounded and never applies a frame, all intermediate topology and
        colour changes keep accumulating and are included in the next send.
        Previously, resetting in acknowledge() caused changes between send-time
        and render-time to be silently dropped.

        When rendering_active is False (tab hidden), all delta computation is
        skipped entirely — including the O(E) edge frozenset which is the most
        expensive part with large fully-connected graphs. Colors/moods are still
        stored so the sim stays current; the sim thread is never blocked.
        """
        with self._data_lock:
            # ── Always update raw display data ────────────────────────────
            self.colors = colors
            self.moods  = moods

            # ── Skip all delta work when tab is not visible ───────────────
            # The O(E) edge frozenset and color diff are pure waste when no
            # render will happen. _prev_edges is left as-is (or None) so when
            # rendering_active becomes True again the full repaint path fires.
            if not self.rendering_active:
                return

            # ── Backpressure gate check ───────────────────────────────────
            ready_to_signal = (self._acked_version == self._graph_version)

            if ready_to_signal:
                # Previous frame was acknowledged — reset accumulated deltas
                # so the new diff starts from the browser's current baseline.
                self._color_delta = {}
                self._edge_delta  = {"added": [], "removed": []}

            # ── Accumulate color delta since last acknowledged send ───────
            # If prev_colors is None this is the very first update.
            prev_c = self._prev_colors
            if prev_c is None:
                pending_color_delta = {str(i): c for i, c in enumerate(colors)}
            else:
                # Merge new changes into any existing pending delta so that
                # skipped frames don't lose their changes.
                pending_color_delta = dict(self._color_delta)
                for i, c in enumerate(colors):
                    if i >= len(prev_c) or c != prev_c[i]:
                        pending_color_delta[str(i)] = c
            self._color_delta = pending_color_delta
            self._prev_colors = colors

            # ── Edge delta: only compute frozenset when actively rendering ─
            # With infinite connection distance and 100 nodes the graph can
            # have thousands of edges. frozenset(graph.edges()) every 10
            # iterations is the most expensive single operation in this method.
            current_edges = frozenset(
                (min(a, b), max(a, b)) for a, b in self.graph.edges()
            )
            prev_e = self._prev_edges
            if prev_e is None:
                self._edge_delta = {
                    "added":   [[a, b] for a, b in current_edges],
                    "removed": [],
                }
            else:
                # Edges that were added then removed between sends cancel out.
                # We track the net change: what's new vs the last acked state.
                prev_added   = frozenset(tuple(e) for e in self._edge_delta["added"])
                prev_removed = frozenset(tuple(e) for e in self._edge_delta["removed"])
                newly_added   = current_edges - prev_e
                newly_removed = prev_e - current_edges
                net_added   = (prev_added   | newly_added)   - newly_removed
                net_removed = (prev_removed | newly_removed) - newly_added
                self._edge_delta = {
                    "added":   [list(e) for e in net_added],
                    "removed": [list(e) for e in net_removed],
                }
            self._prev_edges = current_edges

        if ready_to_signal:
            self._graph_version += 1  # signal Dash: new frame ready

    def acknowledge(self, version: int):
        """
        Called by the Dash callback after reading and dispatching a delta.
        Releases the backpressure gate so the sim thread can signal the next
        frame on its following update() call.

        Delta reset is intentionally NOT done here. It happens at the start
        of the next update() write cycle once this ack is seen. This prevents
        a race where the browser is slow to render: if we reset deltas here
        (at send time) and the browser never applies the frame, the next packet
        would diff from a phantom baseline and lose intermediate changes.
        """
        # _acked_version write is atomic under CPython's GIL — no lock needed,
        # but we take the lock for consistency with the read in update().
        with self._data_lock:
            self._acked_version = version

    def read(self) -> tuple[list, list]:
        """Read colors and moods atomically."""
        with self._data_lock:
            return list(self.colors), list(self.moods)

    def read_delta(self) -> tuple[dict, dict, dict, list]:
        """
        Read both deltas and supporting data atomically.

        Returns:
          color_delta : {str(node_id): color} — only changed nodes
          edge_delta  : {"added": [[a,b],...], "removed": [[a,b],...]}
          moods_by_id : {str(node_id): mood} — full mood map for label rendering
          full_colors : full colors list — needed only for first (full) render

        moods_by_id is a dict so the caller can cheaply extract only the moods
        for nodes present in color_delta without iterating the whole list.
        The contract is that get_latest_series() always returns list copies so
        the sim thread cannot mutate data the Dash thread is reading.
        """
        with self._data_lock:
            moods_by_id = {str(i): m for i, m in enumerate(self.moods)}
            return (
                dict(self._color_delta),
                {
                    "added":   list(self._edge_delta["added"]),
                    "removed": list(self._edge_delta["removed"]),
                },
                moods_by_id,
                list(self.colors),
            )

    def update_iteration(self, i: int):
        """Called from the sim thread each reconstruction event."""
        self.iteration = i  # int write is atomic in CPython — no lock needed

    def update_metrics(self, series: dict):
        """Called from the sim thread after each tracker snapshot."""
        with self._metrics_lock:
            self._metrics = series
        self._metrics_version += 1  # atomic int write

    def read_metrics(self) -> dict:
        """
        Called from the Dash interval callback.

        Returns a deep copy of the metrics snapshot so the Dash thread can
        safely iterate nested lists without racing against the sim thread's
        AgentTracker appending new values to the same list objects.

        Contract: get_latest_series() in AgentTracker must always return a
        fresh snapshot (list copies, not references to live _series buffers).
        deepcopy here enforces isolation regardless of tracker internals.
        """
        import copy
        with self._metrics_lock:
            return copy.deepcopy(self._metrics)


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

    def _adaptive_interval_ms(self) -> int:
        """
        Estimate a sensible Dash polling interval based on the expected
        serialisation cost of one graph update frame.

        Larger sims produce bigger delta payloads (more nodes to diff,
        more edges to track) so we give the server more breathing room
        between polls. Clamped between 500ms (small/fast sims) and
        4000ms (very large sims).

        Formula:
          base = 500ms
          node cost  ≈ 50 bytes each (id, label, color, position, classes)
          edge cost  ≈ 30 bytes each (source, target)
          extra_ms   = estimated_bytes // 1000
          interval   = clamp(base + extra_ms, 500, 4000)
        """
        cfg       = self.state.config
        n_nodes   = cfg.get("num_nodes",  100)
        n_edges   = cfg.get("num_edges",  50)
        estimated = n_nodes * 50 + n_edges * 30
        interval  = min(4000, max(500, 500 + estimated // 1000))
        return interval

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

            # Stores live here so they exist regardless of sidebar state.
            dcc.Store(id={"type": "metrics-store",         "sim": sid}, data={}),
            dcc.Store(id={"type": "graph-version-store",   "sim": sid}, data=-1),
            dcc.Store(id={"type": "metrics-version-store", "sim": sid}, data=-1),
            # Tracks whether the four sidebar charts have been initialised with
            # their full figure skeleton. Once True, updates use extendData
            # (append-only, constant payload) instead of rebuilding figures.
            dcc.Store(id={"type": "plot-init-store",       "sim": sid}, data=False),
            # True when this sim's tab is the active tab. Written by the
            # toggleSimTabs clientside callback on every tab change.
            # sync_and_render and sync_metrics both read this store and return
            # no_update immediately when False — the simulation keeps running
            # in its thread but zero rendering work is done for hidden tabs.
            # When the tab becomes visible again, graph-version-store is reset
            # to -1 so the next tick sends a full repaint of current state.
            dcc.Store(id={"type": "tab-visible-store",     "sim": sid}, data=False),
            # Delta store: holds {node_id: color} for changed nodes only,
            # or the full elements list on first render / topology change.
            # Keys: "type" = "full" or "delta", "payload" = data.
            dcc.Store(id={"type": "graph-delta-store", "sim": sid},
                      data={"type": "none", "payload": {}}),
            dcc.Interval(id={"type": "sim-interval", "sim": sid},
                         interval=self._adaptive_interval_ms(), n_intervals=0),

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
        # Keyed by sim_id string — O(1) lookup in _find_state instead of O(N)
        # linear scan. Insertion order is preserved (Python 3.7+) so iteration
        # over values() still yields sims in launch order.
        self._simulations: dict[str, SimulationState] = {}
        self._sim_counter = 0
        self._lock = threading.Lock()
        # Tracks which sim IDs were finished as of the last tab-bar rebuild.
        # The tab-bar is only re-serialised when this set changes or a new sim
        # is added — not on every shell-interval tick.
        self._finished_sids: frozenset[str] = frozenset()

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
            self._simulations[sid] = state   # O(1) insert, O(1) lookup
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
        self._cb_sync_and_render()   # replaces _cb_sync_stores + _cb_update_graphs
        # _cb_update_iteration removed — iteration is now embedded in the delta
        # packet and applied by the clientside applyGraphPatch function, saving
        # one full server round-trip per interval tick per running simulation.
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
                all_states = list(self._simulations.values())

            # ── Tab bar: only rebuild when something actually changed ──────
            # Check whether the set of finished sims has changed since the
            # last rebuild. If count is unchanged AND no new sim has finished,
            # return no_update for the tab bar — zero bytes cross the wire.
            current_finished = frozenset(s.sim_id for s in all_states if s.finished)
            tabs_changed = (total != known_count) or (current_finished != self._finished_sids)

            if tabs_changed:
                self._finished_sids = current_finished
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
            else:
                tab_children = dash.no_update

            if total <= known_count and not tabs_changed:
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
        Clientside callbacks for tab visibility and graph patch application.
        Also registers a server callback that gates rendering for hidden tabs.
        """
        # Config div visibility — pure CSS toggle, no Python needed.
        # Runs entirely in the browser: zero network traffic, instant response.
        self.app.clientside_callback(
            """
            function(activeTab) {
                return activeTab === '%s' ? {display: 'block'} : {display: 'none'};
            }
            """ % CONFIG_TAB_VALUE,
            Output("config-tab-content", "style"),
            Input("main-tabs", "value"),
        )

        # Tab visibility: named function from assets/clientside.js
        self.app.clientside_callback(
            ClientsideFunction(namespace="ipd_sim", function_name="toggleSimTabs"),
            Output("active-tab-store", "data"),
            Input("main-tabs", "value"),
        )

        # Visibility gate: write tab-visible-store for each sim based on the
        # active tab. When a sim becomes hidden, also reset graph-version-store
        # to -1 so the next time it becomes visible the first tick sends a full
        # repaint — the browser's Cytoscape canvas is stale after being hidden.
        @self.app.callback(
            Output({"type": "tab-visible-store",   "sim": MATCH}, "data"),
            Output({"type": "graph-version-store", "sim": MATCH}, "data",
                   allow_duplicate=True),
            Input("main-tabs", "value"),
            State({"type": "tab-visible-store",    "sim": MATCH}, "id"),
            prevent_initial_call=True,
        )
        def update_visibility(active_tab, store_id):
            sid     = store_id["sim"]
            visible = (active_tab == f"sim-{sid}")
            state   = self._find_state(sid)
            if state is not None:
                if visible:
                    # Becoming visible: activate delta tracking and reset
                    # _prev_edges to None so the next update() call triggers
                    # a clean full-edge snapshot rather than diffing against
                    # a stale pre-hidden baseline.
                    with state._data_lock:
                        state.rendering_active = True
                        state._prev_edges      = None
                        state._prev_colors     = None
                        state._color_delta     = {}
                        state._edge_delta      = {"added": [], "removed": []}
                else:
                    # Becoming hidden: deactivate delta tracking so update()
                    # skips all O(E) frozenset and diff work for this sim.
                    with state._data_lock:
                        state.rendering_active = False
            # When becoming hidden: reset graph-version-store to -1 so the
            # next visible tick forces a full packet instead of a stale delta.
            # When becoming visible: leave graph-version-store alone — the
            # interval callback reads the real current version itself.
            version_reset = -1 if not visible else dash.no_update
            return visible, version_reset

        # Graph patch application: named function from assets/clientside.js
        # "full" packet → replaces all elements (first render / topology change)
        # "delta" packet → patches only changed nodes in-browser (O(changed) payload)
        # Both packet types carry an "iteration" int that applyGraphPatch writes
        # to the iteration-display element, replacing the old _cb_update_iteration
        # server callback entirely.
        # The iteration-display id is passed as the last State so applyGraphPatch
        # can call document.getElementById() directly — no DOM walking, no
        # cross-contamination between simultaneously running simulations.
        self.app.clientside_callback(
            ClientsideFunction(namespace="ipd_sim", function_name="applyGraphPatch"),
            Output({"type": "cytoscape",        "sim": MATCH}, "elements"),
            Output({"type": "cytoscape",        "sim": MATCH}, "zoom"),
            Output({"type": "cytoscape",        "sim": MATCH}, "pan"),
            Input({"type": "graph-delta-store", "sim": MATCH}, "data"),
            State({"type": "cytoscape",         "sim": MATCH}, "elements"),
            State({"type": "cytoscape",         "sim": MATCH}, "zoom"),
            State({"type": "cytoscape",         "sim": MATCH}, "pan"),
            State({"type": "iteration-display", "sim": MATCH}, "id"),
        )

    def _register_sim_visibility(self, sid: str):
        # No longer needed — clientside callback handles all sim divs.
        pass

    def _cb_sync_and_render(self):
        """
        Server side of the delta-encoded graph update pipeline.

        On each interval tick:
          0. Visibility gate — if tab-visible-store is False this sim is not
             on screen. Return no_update immediately. The simulation thread
             keeps running; no packet is assembled or sent. The interval keeps
             ticking so it wakes instantly when the tab is reopened.
             When the tab becomes visible again, update_visibility resets
             graph-version-store to -1, so the next tick here sends a full
             repaint of current state via the last_version == -1 path.
          1. Version check — if _graph_version unchanged, return no_update.
          2. Read delta: {node_id: new_color} for only changed nodes.
          3. Topology change detection: full packet on first render, delta after.
          4. Clientside applyGraphPatch applies the packet in the browser.
        """
        @self.app.callback(
            Output({"type": "graph-delta-store",   "sim": MATCH}, "data"),
            Output({"type": "graph-version-store", "sim": MATCH}, "data"),
            Input({"type": "sim-interval",         "sim": MATCH}, "n_intervals"),
            State({"type": "sim-interval",         "sim": MATCH}, "id"),
            State({"type": "graph-version-store",  "sim": MATCH}, "data"),
            State({"type": "tab-visible-store",    "sim": MATCH}, "data"),
        )
        def sync_and_render(_, interval_id, last_version, is_visible):
            # ── Visibility gate ───────────────────────────────────────────
            if not is_visible:
                return dash.no_update, dash.no_update
            sid   = interval_id["sim"]
            state = self._find_state(sid)
            if state is None or state.paused:
                return dash.no_update, dash.no_update
            current_version = state._graph_version
            if current_version == last_version:
                return dash.no_update, dash.no_update

            color_delta, edge_delta, moods_by_id, full_colors = state.read_delta()

            if last_version == -1:
                # First render only — send complete elements list so the
                # browser has a baseline to patch against from this point on.
                # full moods list is reconstructed from moods_by_id here.
                full_moods = [moods_by_id.get(str(i)) for i in range(len(full_colors))]
                elements = CytoscapeRenderer.elements(
                    state.graph, full_colors, state.dimensions,
                    moods=full_moods, positions=state.positions,
                )
                packet = {"type": "full", "payload": elements,
                          "iteration": state.iteration}
            else:
                # Every subsequent update uses deltas for both nodes and edges.
                # mood_delta contains only the mood values for nodes that are
                # already in color_delta — no need to send all N moods every tick.
                # The clientside patch function reads mood from mood_delta[id]
                # rather than indexing into the full list.
                mood_delta = {nid: moods_by_id[nid]
                              for nid in color_delta if nid in moods_by_id}
                packet = {
                    "type":          "delta",
                    "payload":       color_delta,
                    "mood_delta":    mood_delta,
                    "iteration":     state.iteration,
                    "edges_added":   edge_delta["added"],
                    "edges_removed": edge_delta["removed"],
                }

            # Acknowledge this version — releases the backpressure gate so
            # the sim thread can write the next frame. Also resets deltas.
            state.acknowledge(current_version)
            return packet, current_version

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
        Checks tab visibility first, then _metrics_version before touching
        the metrics dict at all. Hidden tabs return no_update immediately —
        the sim thread keeps appending to its series buffers, and the full
        up-to-date metrics are sent as one batch the first time the tab is
        opened (metrics-version-store diverges while hidden, so it always
        sends on the next visible tick).
        """
        @self.app.callback(
            Output({"type": "metrics-store",          "sim": MATCH}, "data"),
            Output({"type": "metrics-version-store",  "sim": MATCH}, "data"),
            Input({"type": "sim-interval",            "sim": MATCH}, "n_intervals"),
            State({"type": "sim-interval",            "sim": MATCH}, "id"),
            State({"type": "metrics-version-store",   "sim": MATCH}, "data"),
            State({"type": "tab-visible-store",       "sim": MATCH}, "data"),
        )
        def sync_metrics(_, interval_id, last_version, is_visible):
            # ── Visibility gate ───────────────────────────────────────────
            if not is_visible:
                return dash.no_update, dash.no_update
            sid   = interval_id["sim"]
            state = self._find_state(sid)
            if state is None or state.paused:
                return dash.no_update, dash.no_update
            current_version = state._metrics_version
            if current_version == last_version:
                return dash.no_update, dash.no_update
            metrics = state.read_metrics()
            if not metrics:
                return dash.no_update, dash.no_update
            return metrics, current_version

    def _cb_update_plots(self):
        """
        Update the four sidebar charts incrementally using extendData.

        On first data arrival the figures are built once with their full
        skeleton (axes, layout, empty traces in the right order/colours).
        Every subsequent update appends only the newest [x, y] point per
        trace via the extendData property — a constant-size payload
        regardless of how many rounds the simulation has run.

        Previously this rebuilt four complete go.Figure objects containing
        the entire time-series on every metrics update, with payload size
        growing linearly without bound over the simulation lifetime.
        """
        # Trace order must match _CLASS_COLORS iteration order — fixed here
        # so extendData indices are stable.
        _CLASSES  = ["MoodySARSA", "SARSA", "TFT", "WSLS"]
        _N_TRACES = len(_CLASSES)

        def _initial_figure(y_title: str = "", y_range: list | None = None) -> go.Figure:
            """Build the figure skeleton with empty traces in correct order."""
            fig = go.Figure()
            for cls in _CLASSES:
                fig.add_trace(go.Scatter(
                    x=[], y=[],
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

        def _extend_payload(metrics: dict, metric: str) -> tuple[dict, list]:
            """
            Build an extendData payload for one chart.
            Returns ({x: [...], y: [...]}, trace_indices) where each list has
            one element per trace — the single newest data point.
            """
            rounds  = metrics.get("rounds", [])
            classes = metrics.get(metric, {})
            if not rounds:
                return {}, []
            latest_x = rounds[-1]
            xs = [latest_x] * _N_TRACES
            ys = [classes.get(cls, [None])[-1] if classes.get(cls) else None
                  for cls in _CLASSES]
            return (
                {"x": [[x] for x in xs], "y": [[y] for y in ys]},
                list(range(_N_TRACES)),
            )

        @self.app.callback(
            Output({"type": "plot-score",        "sim": MATCH}, "figure"),
            Output({"type": "plot-connectivity", "sim": MATCH}, "figure"),
            Output({"type": "plot-cooperation",  "sim": MATCH}, "figure"),
            Output({"type": "plot-mood",         "sim": MATCH}, "figure"),
            Output({"type": "plot-score",        "sim": MATCH}, "extendData"),
            Output({"type": "plot-connectivity", "sim": MATCH}, "extendData"),
            Output({"type": "plot-cooperation",  "sim": MATCH}, "extendData"),
            Output({"type": "plot-mood",         "sim": MATCH}, "extendData"),
            Output({"type": "sidebar-waiting",   "sim": MATCH}, "style"),
            Output({"type": "plot-init-store",   "sim": MATCH}, "data"),
            Input({"type": "metrics-store",      "sim": MATCH}, "data"),
            State({"type": "plot-init-store",    "sim": MATCH}, "data"),
        )
        def update_plots(metrics, already_initialised):
            no_data_style = {"fontSize": "0.78rem", "color": "#868e96",
                             "fontStyle": "italic", "marginBottom": "8px"}
            hidden        = {"display": "none"}
            no_ext        = dash.no_update  # placeholder for extendData outputs

            if not metrics or not metrics.get("rounds"):
                return (
                    _empty_figure(), _empty_figure(),
                    _empty_figure(), _empty_figure(),
                    no_ext, no_ext, no_ext, no_ext,
                    no_data_style, False,
                )

            if not already_initialised:
                # First data arrival — build figure skeletons and populate all
                # existing points so the charts are immediately correct.
                # After this, only extendData is used (figure outputs = no_update).
                rounds   = metrics["rounds"]
                fig_score = _initial_figure(y_title="Avg payoff")
                fig_conn  = _initial_figure(y_title="Norm. degree", y_range=[0, 1])
                fig_coop  = _initial_figure(y_title="Coop. rate",   y_range=[0, 1])
                fig_mood  = _initial_figure(y_title="Mood (0–100)", y_range=[0, 100])
                for i, cls in enumerate(_CLASSES):
                    for fig, metric in [
                        (fig_score, "score"), (fig_conn, "connectivity"),
                        (fig_coop, "cooperation"), (fig_mood, "mood"),
                    ]:
                        vals = metrics.get(metric, {}).get(cls, [])
                        fig.data[i].x = rounds
                        fig.data[i].y = vals
                return (
                    fig_score, fig_conn, fig_coop, fig_mood,
                    no_ext, no_ext, no_ext, no_ext,
                    hidden, True,
                )

            # Already initialised — append only the newest point via extendData.
            # Payload is constant size: 4 traces × 1 point × 4 charts.
            # No go.Figure objects are constructed at all.
            ext_score = _extend_payload(metrics, "score")
            ext_conn  = _extend_payload(metrics, "connectivity")
            ext_coop  = _extend_payload(metrics, "cooperation")
            ext_mood  = _extend_payload(metrics, "mood")
            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                ext_score, ext_conn, ext_coop, ext_mood,
                hidden, True,
            )

    # _cb_sidebar removed — analysis panel is now a permanent left panel
    # in the two-column layout, no toggle needed.

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_state(self, sid: str) -> SimulationState | None:
        # O(1) dict lookup — previously an O(N) linear scan under a lock.
        with self._lock:
            return self._simulations.get(sid)


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