"""
gui_app.py
──────────
Configuration UI. Builds the config page layout and registers its callbacks.
All routing/tab logic lives in visualization.SimulationApp.

Layout design
─────────────
Two-column grid:
  Left  – simulation parameters (nodes, edges, games, thresholds, sliders)
  Right – agent type weights with live normalised % indicators

Each slider row has an inline number input so the user can type precise values.
Agent weight sliders show both raw weight and normalised population percentage.
"""

import json
import math
import random
import threading

import dash
import dash_bootstrap_components as dbc
import networkx as nx
from dash import Input, Output, State, ALL, ctx, dcc, html

from flask_server import get_server
from simulation_core import (
    AGENT_COLOR_MAP,
    COLOR_AGENT_MAP,
    GraphFactory,
    Simulation,
    get_pool,
)
from visualization import SimulationApp


# ──────────────────────────────────────────────────────────────────────────────
# Module-level shared state
# ──────────────────────────────────────────────────────────────────────────────
# These dicts represent the CURRENT config-page state (what the user is
# looking at). They are read-only after a simulation is launched — each
# launch takes a deep copy so concurrent sims never share references.

_node_colors: dict[int, str] = {}
_node_positions: dict[int, dict] = {}
_COLOR_CYCLE = ["red", "green", "orange", "pink", "white", "yellow"]

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

# Inline style helpers
_CARD = {"borderRadius": "8px", "padding": "16px", "backgroundColor": "#f8f9fa",
         "marginBottom": "12px"}
_LABEL_STYLE = {"fontWeight": "600", "fontSize": "0.85rem", "marginBottom": "2px"}
_INPUT_SMALL = {"width": "80px", "height": "32px", "fontSize": "0.85rem",
                "padding": "2px 6px", "borderRadius": "4px", "border": "1px solid #ced4da"}
_PCT_BADGE = {"fontSize": "0.75rem", "padding": "2px 7px", "borderRadius": "10px",
              "backgroundColor": "#dee2e6", "color": "#495057",
              "whiteSpace": "nowrap", "minWidth": "52px", "textAlign": "center"}


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _labeled_number(label: str, input_id: str, value, min_=None, step=1,
                    width="110px") -> html.Div:
    """A compact labelled number input."""
    kwargs = {"id": input_id, "type": "number", "value": value,
              "step": step, "style": {**_INPUT_SMALL, "width": width}}
    if min_ is not None:
        kwargs["min"] = min_
    return html.Div([
        html.Div(label, style=_LABEL_STYLE),
        dcc.Input(**kwargs),
    ], className="mb-2")


def _slider_with_input(label: str, slider_id: str, input_id: str,
                       min_=0.0, max_=1.0, step=0.01, value=0.0,
                       marks=None) -> html.Div:
    """
    A row with a label, a slider, and a small number input that stay in sync.
    Used for both parameter sliders and agent weight sliders.
    """
    if marks is None:
        marks = {min_: str(min_), max_: str(max_)}
    return html.Div([
        html.Div(label, style=_LABEL_STYLE),
        html.Div([
            html.Div(
                dcc.Slider(
                    id=slider_id, min=min_, max=max_, step=step, value=value,
                    marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                style={"flex": "1", "minWidth": "0"},
            ),
            dcc.Input(
                id=input_id, type="number", value=value,
                min=min_, max=max_, step=step,
                style={**_INPUT_SMALL, "marginLeft": "8px", "flexShrink": "0"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], className="mb-2")


def _agent_weight_row(agent: str, color: str) -> html.Div:
    """
    One row in the agent weights panel:
      [●  AgentName]  [────slider────]  [input]  [XX%]
    The percentage badge is updated by a callback.
    """
    dot = html.Span(style={
        "display": "inline-block", "width": "10px", "height": "10px",
        "borderRadius": "50%", "backgroundColor": color,
        "border": "1px solid #aaa", "marginRight": "6px", "flexShrink": "0",
    })
    return html.Div([
        # Label
        html.Div(
            [dot, html.Span(agent, style={"fontSize": "0.82rem", "fontWeight": "600"})],
            style={"display": "flex", "alignItems": "center", "marginBottom": "2px"},
        ),
        # Slider + input + badge
        html.Div([
            html.Div(
                dcc.Slider(
                    id={"type": "weight-slider", "agent": agent},
                    min=0, max=1, step=0.01,
                    value=1 if agent == "MoodySARSAAgent" else 0,
                    marks={0: "0", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                style={"flex": "1", "minWidth": "0"},
            ),
            dcc.Input(
                id={"type": "weight-input", "agent": agent},
                type="number", min=0, max=1, step=0.01,
                value=1 if agent == "MoodySARSAAgent" else 0,
                style={**_INPUT_SMALL, "marginLeft": "8px", "flexShrink": "0"},
            ),
            html.Span(
                "0%",
                id={"type": "weight-pct", "agent": agent},
                style={**_PCT_BADGE, "marginLeft": "6px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], className="mb-2")


def _sarsa_panel(div_id: str, title: str,
                 eps_id: str, alpha_id: str, gamma_id: str) -> html.Div:
    return html.Div([
        html.Div(title, style={**_LABEL_STYLE, "fontSize": "0.9rem",
                               "marginBottom": "6px", "color": "#495057"}),
        html.Div([
            _labeled_number("ε Epsilon", eps_id,   0.1,  step=0.01, width="80px"),
            _labeled_number("α Alpha",   alpha_id, 0.1,  step=0.01, width="80px"),
            _labeled_number("γ Gamma",   gamma_id, 0.95, step=0.01, width="80px"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ], id=div_id, style={"display": "none", "marginTop": "8px",
                         "padding": "10px", "backgroundColor": "#e9ecef",
                         "borderRadius": "6px"})


# ──────────────────────────────────────────────────────────────────────────────
# Layout builder
# ──────────────────────────────────────────────────────────────────────────────

class ConfigLayoutBuilder:

    _AGENTS = list(AGENT_COLOR_MAP.keys())

    @classmethod
    def build(cls) -> html.Div:
        return html.Div([
            # ── Page title + action buttons ───────────────────────────────
            html.Div([
                html.H4("Simulation Configuration",
                        style={"margin": "0", "fontWeight": "700"}),
                html.Div([
                    dbc.Button("Preview Graph", id="overview-btn",
                               color="primary", size="sm"),
                    dbc.Button("▶ Start Simulation", id="run-sim-btn",
                               color="success", size="sm", className="ms-2",
                               disabled=True,
                               style={"opacity": "0.5", "cursor": "not-allowed"}),
                ]),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px",
                      "paddingBottom": "8px", "borderBottom": "1px solid #dee2e6"}),

            html.Div(id="config-output", className="mb-2"),

            # ── Two-column body ───────────────────────────────────────────
            dbc.Row([
                # LEFT column — sim parameters
                dbc.Col([
                    html.Div([
                        html.Div("Simulation Parameters",
                                 style={**_LABEL_STYLE, "fontSize": "0.95rem",
                                        "marginBottom": "10px", "color": "#343a40"}),

                        _labeled_number("Simulation Name", "sim-name-input", "",
                                        step=None, width="200px"),

                        html.Div([
                            _labeled_number("Nodes",    "num-nodes", 100, min_=1),
                            _labeled_number("Edges",    "num-edges",  50, min_=0),
                            _labeled_number("Games",    "num-games", 50_000, min_=1, width="120px"),
                        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

                        _slider_with_input(
                            "Avg Betrayal Threshold",
                            "betrayal-threshold-slider", "betrayal-threshold",
                            min_=0.0, max_=5.0, step=0.1, value=2.5,
                            marks={0: "0", 2.5: "2.5", 5: "5"},
                        ),
                        _slider_with_input(
                            "Max Connection Distance",
                            "max-dist-slider", "max-dist",
                            min_=50, max_=100_001, step=50, value=350,
                            marks={50: "50", 350: "350", 100_001: "∞"},
                        ),
                        _slider_with_input(
                            "% Reconnection (R%)",
                            "reconnect-pct", "reconnect-pct-input",
                            min_=0.0, max_=1.0, step=0.01, value=0.2,
                            marks={0: "0%", 0.5: "50%", 1: "100%"},
                        ),

                        html.Div([
                            html.Div("Edge Creation Method", style=_LABEL_STYLE),
                            dbc.RadioItems(
                                id="forgiveness-mode",
                                options=[
                                    {"label": "WIPE", "value": "WIPE"},
                                    {"label": "POP",  "value": "POP"},
                                ],
                                value="WIPE", inline=True,
                                style={"fontSize": "0.85rem"},
                            ),
                        ], className="mb-2"),

                    ], style=_CARD),
                ], md=5),

                # RIGHT column — agent weights
                dbc.Col([
                    html.Div([
                        html.Div("Agent Type Weights",
                                 style={**_LABEL_STYLE, "fontSize": "0.95rem",
                                        "marginBottom": "10px", "color": "#343a40"}),
                        *[_agent_weight_row(a, AGENT_COLOR_MAP[a])
                          for a in cls._AGENTS],

                        html.Hr(style={"margin": "10px 0"}),

                        # SARSA panels (hidden until weight > 0)
                        _sarsa_panel("sarsa-config", "SARSA Parameters",
                                     "sarsa-epsilon", "sarsa-alpha", "sarsa-gamma"),
                        _sarsa_panel("moody-config", "Moody SARSA Parameters",
                                     "moody-epsilon", "moody-alpha", "moody-gamma"),
                    ], style=_CARD),
                ], md=7),
            ]),

            # ── Preview graph (framed card, only visible when populated) ─
            html.Div([
                html.Div([
                    html.Span("Graph Preview",
                              style={"fontWeight": "600", "fontSize": "0.9rem",
                                     "color": "#495057"}),
                    html.Span(" — click any node to cycle its agent type",
                              style={"fontSize": "0.78rem", "color": "#868e96"}),
                ], style={"marginBottom": "8px"}),
                html.Div(id="cytoscape-container"),
            ], id="preview-card", style={
                "display": "none",
                "marginTop": "14px",
                "border": "1px solid #dee2e6",
                "borderRadius": "8px",
                "padding": "14px",
                "backgroundColor": "#ffffff",
            }),
            html.Div(id="refresh-div", style={"display": "none"}),

        ], id="config-content-mount", style={"padding": "16px"})


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class ConfigCallbacks:

    def __init__(self, app: dash.Dash, sim_app: SimulationApp):
        self._app = app
        self._sim_app = sim_app
        self._register()

    def _register(self):
        self._sync_weight_sliders_and_inputs()
        self._update_pct_badges()
        self._toggle_sarsa_panels()
        self._sync_param_sliders()
        self._preview_graph()
        self._cycle_node_color()
        self._save_positions()
        self._launch_simulation()

    # ── Weight slider ↔ input sync ────────────────────────────────────────

    def _sync_weight_sliders_and_inputs(self):
        """
        Sync weight sliders and their number inputs without creating a cycle.

        Strategy: one callback, two inputs, ctx.triggered_id decides direction.
        - Slider moved  → update the input (user dragged the slider)
        - Input changed → update the slider (user typed a value)
        Whichever side did NOT change gets dash.no_update so the cycle stops.

        With pattern-matched ALL outputs, no_update must be returned as a
        full list — we build one from the unchanged side's current values.
        """
        @self._app.callback(
            Output({"type": "weight-slider", "agent": ALL}, "value"),
            Output({"type": "weight-input",  "agent": ALL}, "value"),
            Input({"type": "weight-slider",  "agent": ALL}, "value"),
            Input({"type": "weight-input",   "agent": ALL}, "value"),
            prevent_initial_call=True,
        )
        def sync_weights(slider_vals, input_vals):
            triggered = ctx.triggered_id
            if not triggered:
                return dash.no_update, dash.no_update
            n = len(slider_vals)
            if triggered.get("type") == "weight-slider":
                # Slider moved → push to inputs, leave sliders alone
                return [dash.no_update] * n, slider_vals
            else:
                # Input typed → push to sliders, leave inputs alone
                return input_vals, [dash.no_update] * n

    # ── Normalised % badges ───────────────────────────────────────────────

    def _update_pct_badges(self):
        @self._app.callback(
            Output({"type": "weight-pct", "agent": ALL}, "children"),
            Input({"type": "weight-slider", "agent": ALL}, "value"),
        )
        def update_pcts(weights):
            total = sum(w or 0 for w in weights) or 1
            return [f"{(w or 0) / total * 100:.0f}%" for w in weights]

    # ── SARSA panel toggle ────────────────────────────────────────────────

    def _toggle_sarsa_panels(self):
        agents = list(AGENT_COLOR_MAP.keys())

        @self._app.callback(
            Output("sarsa-config", "style"),
            Output("moody-config", "style"),
            Input({"type": "weight-slider", "agent": ALL}, "value"),
            State({"type": "weight-slider", "agent": ALL}, "id"),
        )
        def toggle(weights, ids):
            wmap = {d["agent"]: (w or 0) for d, w in zip(ids, weights)}
            show = {"display": "block", "marginTop": "8px", "padding": "10px",
                    "backgroundColor": "#e9ecef", "borderRadius": "6px"}
            hide = {"display": "none"}
            return (
                show if wmap.get("SARSAAgent", 0) > 0 else hide,
                show if wmap.get("MoodySARSAAgent", 0) > 0 else hide,
            )

    # ── Param slider ↔ input sync ─────────────────────────────────────────

    def _sync_param_sliders(self):
        """
        Keep each parameter slider and its paired number input in sync.
        Single callback per pair, ctx.triggered_id breaks the cycle:
        whichever component fired pushes its value to the other; the
        component that fired returns no_update so it isn't re-triggered.
        """
        pairs = [
            ("betrayal-threshold-slider", "betrayal-threshold"),
            ("max-dist-slider",           "max-dist"),
            ("reconnect-pct",             "reconnect-pct-input"),
        ]
        for slider_id, input_id in pairs:
            @self._app.callback(
                Output(slider_id, "value"),
                Output(input_id,  "value"),
                Input(slider_id,  "value"),
                Input(input_id,   "value"),
                prevent_initial_call=True,
            )
            def sync_pair(sv, iv, _sid=slider_id, _iid=input_id):
                triggered = ctx.triggered_id
                if triggered == _sid:
                    return dash.no_update, sv   # slider moved → update input
                return iv, dash.no_update       # input typed → update slider

    # ── Preview graph ─────────────────────────────────────────────────────

    def _preview_graph(self):
        @self._app.callback(
            Output("cytoscape-container", "children"),
            Output("run-sim-btn",  "disabled"),
            Output("run-sim-btn",  "style"),
            Output("preview-card", "style"),
            Input("overview-btn", "n_clicks"),
            State("num-nodes", "value"),
            State({"type": "weight-slider", "agent": ALL}, "value"),
            State({"type": "weight-slider", "agent": ALL}, "id"),
        )
        def preview(n_clicks, num_nodes, weights, ids):
            if not n_clicks:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            import dash_cytoscape as cyto

            wmap = {d["agent"]: (w or 0) for d, w in zip(ids, weights)}
            total = sum(wmap.values()) or 1
            color_list = [AGENT_COLOR_MAP[a] for a in wmap]
            norm = [wmap[a] / total for a in wmap]

            for i in range(num_nodes):
                _node_colors[i] = random.choices(color_list, norm)[0]

            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            dims = int(math.sqrt(num_nodes))

            card_visible = {
                "display": "block", "marginTop": "14px",
                "border": "1px solid #dee2e6", "borderRadius": "8px",
                "padding": "14px", "backgroundColor": "#ffffff",
            }
            return (
                cyto.Cytoscape(
                    id="overview-graph",
                    layout={"name": "preset"},
                    elements=self._make_elements(graph, dims),
                    style={"width": "100%", "height": "480px"},
                    stylesheet=CYTO_STYLESHEET,
                    userZoomingEnabled=False,
                ),
                False,
                {"opacity": "1", "cursor": "pointer"},
                card_visible,
            )

    # ── Cycle node color on click ─────────────────────────────────────────

    def _cycle_node_color(self):
        @self._app.callback(
            Output("overview-graph", "elements"),
            Input("overview-graph", "tapNodeData"),
            State("num-nodes", "value"),
            State("overview-graph", "elements"),
        )
        def cycle(node_data, num_nodes, current_elements):
            if not node_data:
                return dash.no_update
            node_id = int(node_data["id"])
            current = _node_colors.get(node_id, "orange")
            idx = _COLOR_CYCLE.index(current) if current in _COLOR_CYCLE else -1
            _node_colors[node_id] = _COLOR_CYCLE[(idx + 1) % len(_COLOR_CYCLE)]
            self._sync_positions(current_elements)
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            dims = int(math.sqrt(num_nodes))
            return self._make_elements(graph, dims, use_saved=True)

    # ── Save positions on mouseover ───────────────────────────────────────

    def _save_positions(self):
        @self._app.callback(
            Output("refresh-div", "children"),
            Input("overview-graph", "mouseoverNodeData"),
            State("overview-graph", "elements"),
        )
        def save(_, elements):
            if not elements:
                return dash.no_update
            self._sync_positions(elements)
            return f"Positions saved: {len(_node_positions)}"

    # ── Launch simulation ─────────────────────────────────────────────────

    def _launch_simulation(self):
        sarsa_states = [
            State("sarsa-epsilon", "value"), State("sarsa-alpha",  "value"),
            State("sarsa-gamma",   "value"), State("moody-epsilon", "value"),
            State("moody-alpha",   "value"), State("moody-gamma",   "value"),
        ]

        @self._app.callback(
            Output("config-output",       "children"),
            Output("cytoscape-container", "children",  allow_duplicate=True),
            Output("run-sim-btn",         "disabled",  allow_duplicate=True),
            Output("run-sim-btn",         "style",     allow_duplicate=True),
            Output("preview-card",        "style",     allow_duplicate=True),
            Input("run-sim-btn", "n_clicks"),
            State("overview-graph", "elements"),
            State("sim-name-input",     "value"),
            State("num-nodes",          "value"),
            State("num-edges",          "value"),
            State("num-games",          "value"),
            State("betrayal-threshold", "value"),
            State("reconnect-pct",      "value"),
            State("max-dist",           "value"),
            State("forgiveness-mode",   "value"),
            State({"type": "weight-slider", "agent": ALL}, "value"),
            State({"type": "weight-slider", "agent": ALL}, "id"),
            *sarsa_states,
            prevent_initial_call=True,
        )
        def launch(
            n_clicks, elements, sim_name,
            num_nodes, num_edges, num_games,
            betrayal_thresh, reconnect_pct, max_dist, forgiveness_mode,
            weights, weight_ids,
            sarsa_eps, sarsa_alpha, sarsa_gamma,
            moody_eps, moody_alpha, moody_gamma,
        ):
            if not n_clicks:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            self._sync_positions(elements)

            assignment = {
                str(node): COLOR_AGENT_MAP.get(color, "WSLSAgent")
                for node, color in _node_colors.items()
            }

            clean_name = (sim_name or "").strip()

            # Deep-copy positions and colors so this sim's config is fully
            # isolated — mutating _node_positions later won't affect it.
            import copy
            config = {
                "sim_name":                    clean_name,
                "num_nodes":                   num_nodes,
                "num_edges":                   num_edges,
                "node_positions":              copy.deepcopy(_node_positions),
                "node_colors":                 {str(k): v for k, v in _node_colors.items()},
                "agent_assignment":            assignment,
                "num_games_per_pair":          num_games,
                "percent_reconnection":        reconnect_pct,
                "max_connection_distance":     float(max_dist),
                "forgiveness_mode":            forgiveness_mode,
                "average_considered_betrayal": betrayal_thresh,
                "SARSAAgent_params":    {"epsilon": sarsa_eps,  "alpha": sarsa_alpha,
                                         "gamma": sarsa_gamma},
                "MoodySARSAAgent_params": {"epsilon": moody_eps, "alpha": moody_alpha,
                                           "gamma": moody_gamma},
            }

            with open("params.json", "w") as f:
                json.dump(config, f, indent=4)

            graph = GraphFactory.create(num_nodes)
            graph = GraphFactory.populate_edges(
                graph, num_edges, num_nodes, _node_positions, float(max_dist)
            )
            dims = int(math.sqrt(num_nodes))
            initial_colors = [_node_colors.get(i, "orange") for i in range(num_nodes)]

            state = self._sim_app.attach_simulation(
                graph, dims, dict(_node_positions), initial_colors,
                sim_name=clean_name,
                config=config,
            )

            sim = Simulation(state, graph, config)
            # Submit to the shared thread pool — manages all sim threads
            get_pool().submit(sim)

            label = f'"{clean_name}"' if clean_name else f"Sim {state.sim_id}"
            disabled_style = {"opacity": "0.5", "cursor": "not-allowed"}
            card_hidden = {"display": "none", "marginTop": "14px",
                           "border": "1px solid #dee2e6", "borderRadius": "8px",
                           "padding": "14px", "backgroundColor": "#ffffff"}
            return (
                dbc.Alert(f"✓ {label} started.", color="success",
                          className="mt-1 py-2", style={"fontSize": "0.85rem"}),
                None,
                True,
                disabled_style,
                card_hidden,
            )

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _sync_positions(elements: list):
        if not elements:
            return
        for el in elements:
            if "position" in el and "id" in el.get("data", {}):
                try:
                    _node_positions[int(el["data"]["id"])] = el["position"]
                except (ValueError, KeyError):
                    pass

    @staticmethod
    def _make_elements(graph: nx.Graph, dims: int, use_saved: bool = False) -> list:
        offset = 12.5
        elements = []
        for node in graph.nodes():
            color = _node_colors.get(node, "orange")
            if use_saved and node in _node_positions:
                position = _node_positions[node]
            else:
                row, col = divmod(node, dims)
                position = {
                    "x": col * 125 + (row % 2) * offset,
                    "y": row * 125 + (col % 2) * offset,
                }
                _node_positions[node] = position
            elements.append({
                "data": {"id": str(node), "label": f"Node {node}", "color": color},
                "position": position,
                "classes": color,
            })
        return elements


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def create_app() -> SimulationApp:
    server = get_server()
    config_layout = ConfigLayoutBuilder.build()
    sim_app = SimulationApp(server, config_layout)
    ConfigCallbacks(sim_app.app, sim_app)
    return sim_app


if __name__ == "__main__":
    gui = create_app()
    gui.app.run(debug=True)