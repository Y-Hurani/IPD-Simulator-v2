"""
simulation_core.py
──────────────────
Domain logic for the Iterated Prisoner's Dilemma simulation.
No Dash / Flask imports live here.

Parallelism model
─────────────────
Each Simulation runs in its own daemon thread via SimulationThreadPool.
Threads are concurrent (Python GIL limits true CPU parallelism for pure-Python
loops), but I/O waits, sleep() calls, and threading.Event.wait() all release
the GIL, so multiple sims do make real forward progress simultaneously.

Each simulation is fully isolated:
  - Its own graph, agents, env, reconstructor — no shared mutable objects
  - Its own SimulationState with a threading.Lock around color/mood writes
  - Its own CSV output paths (timestamped, uniquely named)
  - No module-level globals touched during run()
"""

import math
import random
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

from Agent import Agent
from CooperativeAgent import CooperativeAgent
from TFTAgent import TFTAgent
from WSLSAgent import WSLSAgent
from DefectingAgent import DefectingAgent
from SARSAAgent import SARSAAgent
from MoodySARSAAgent import MoodySARSAAgent
from AgentTracker import AgentTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_CLASS_MAP: dict[str, type] = {
    "MoodySARSAAgent": MoodySARSAAgent,
    "SARSAAgent": SARSAAgent,
    "CooperativeAgent": CooperativeAgent,
    "DefectingAgent": DefectingAgent,
    "TFTAgent": TFTAgent,
    "WSLSAgent": WSLSAgent,
}

AGENT_COLOR_MAP: dict[str, str] = {
    "SARSAAgent": "yellow",
    "MoodySARSAAgent": "white",
    "CooperativeAgent": "green",
    "DefectingAgent": "red",
    "TFTAgent": "pink",
    "WSLSAgent": "orange",
}

# Inverse: color → agent type name (used when reading back node colours from GUI)
COLOR_AGENT_MAP: dict[str, str] = {v: k for k, v in AGENT_COLOR_MAP.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euclidean_distance(pos_a: dict, pos_b: dict) -> float:
    return math.sqrt((pos_b["x"] - pos_a["x"]) ** 2 + (pos_b["y"] - pos_a["y"]) ** 2)


def mood_to_hex_color(mood: int) -> str:
    """Map a mood value (1–100) to a greyscale hex string."""
    mood = max(1, min(100, mood))
    intensity = int((mood / 100) * 255)
    h = f"{intensity:02x}"
    return f"#{h}{h}{h}"


def agent_display_color(agent: Agent) -> str:
    if isinstance(agent, CooperativeAgent):
        return "green"
    if isinstance(agent, MoodySARSAAgent):
        return mood_to_hex_color(agent.mood)
    if isinstance(agent, SARSAAgent):
        return "yellow"
    if isinstance(agent, TFTAgent):
        return "pink"
    if isinstance(agent, DefectingAgent):
        return "red"
    return "orange"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PrisonersDilemmaEnvironment:
    """Tracks payoffs and cooperation/defection totals for one simulation run."""

    PAYOFF_TABLE = {
        (1, 1): (3, 3),
        (1, 0): (0, 5),
        (0, 1): (5, 0),
        (0, 0): (1, 1),
    }

    def __init__(self, n_agents: int, n_states: int = 10):
        self.n_agents = n_agents
        self.n_states = n_states
        self.total = [0, 0]  # [defections, cooperations]

    def reset(self):
        self.total = [0, 0]

    def step(self, action_a: int, action_b: int) -> tuple[int, int]:
        reward_a, reward_b = self.PAYOFF_TABLE[(action_a, action_b)]
        if action_a == 1 and action_b == 1:
            self.total[1] += 2
        elif action_a == 0 and action_b == 0:
            self.total[0] += 2
        else:
            self.total[0] += 1
        return reward_a, reward_b


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

class GraphFactory:
    """Builds and mutates NetworkX graphs for the simulation."""

    @staticmethod
    def create(num_nodes: int) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        return g

    @staticmethod
    def populate_edges(
        graph: nx.Graph,
        num_edges: int,
        num_nodes: int,
        positions: dict,
        max_distance: float,
    ) -> nx.Graph:
        attempts = 0
        max_attempts = num_edges * 100  # avoid infinite loop on sparse grids
        while graph.number_of_edges() < num_edges and attempts < max_attempts:
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            if a == b or graph.has_edge(a, b):
                attempts += 1
                continue
            if euclidean_distance(positions[a], positions[b]) < max_distance:
                graph.add_edge(a, b)
            attempts += 1
        return graph

    @staticmethod
    def max_degrees(graph: nx.Graph, positions: dict, max_distance: float) -> dict[int, int]:
        """Return the maximum number of reachable neighbours for every node."""
        result = {}
        nodes = list(graph.nodes())
        for a in nodes:
            result[a] = sum(
                1 for b in nodes
                if b != a and euclidean_distance(positions[a], positions[b]) < max_distance
            )
        return result


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

class AgentFactory:
    """Creates agent instances from an explicit assignment or weighted random."""

    @staticmethod
    def from_assignment(
        n_agents: int,
        assignment: dict[str, str],
        n_states: int,
        n_actions: int,
    ) -> list[Agent]:
        agents = []
        for agent_id in range(n_agents):
            cls = AGENT_CLASS_MAP[assignment[str(agent_id)]]
            agents.append(cls(id=agent_id, n_states=n_states, n_actions=n_actions, n_agents=n_agents))
        return agents

    @staticmethod
    def from_weights(
        n_agents: int,
        weights: dict[str, float],
        n_states: int,
        n_actions: int,
    ) -> list[Agent]:
        total = sum(weights.values()) or 1
        types = list(AGENT_CLASS_MAP.keys())
        probs = [weights.get(t, 0) / total for t in types]
        agents = []
        for agent_id in range(n_agents):
            cls = AGENT_CLASS_MAP[random.choices(types, probs)[0]]
            agents.append(cls(id=agent_id, n_states=n_states, n_actions=n_actions, n_agents=n_agents))
        return agents


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

class GameRunner:
    """Executes a single IPD game between two agents."""

    def __init__(self, env: PrisonersDilemmaEnvironment):
        self.env = env

    def play(self, agent_a: Agent, agent_b: Agent, fixed: int = 0):
        state_a = agent_a.mood if isinstance(agent_a, MoodySARSAAgent) else 50
        state_b = agent_b.mood if isinstance(agent_b, MoodySARSAAgent) else 50

        action_a = agent_a.choose_action(state_b, agent_b.id, fixed)
        action_b = agent_b.choose_action(state_a, agent_a.id, fixed)

        reward_a, reward_b = self.env.step(action_a, action_b)

        next_state_a = (
            max(0, min(99, agent_a.mood + int(reward_a - agent_a.prev_omegas[agent_b.id])))
            if isinstance(agent_a, MoodySARSAAgent) else 50
        )
        next_state_b = (
            max(0, min(99, agent_b.mood + int(reward_b - agent_b.prev_omegas[agent_a.id])))
            if isinstance(agent_b, MoodySARSAAgent) else 50
        )

        next_action_a = agent_a.choose_action(next_state_a, agent_b.id, fixed)
        next_action_b = agent_b.choose_action(next_state_b, agent_a.id, fixed)

        agent_a.after_game_function(state_a, action_a, next_state_a, next_action_a, reward_a, reward_b, agent_b.id)
        agent_b.after_game_function(state_b, action_b, next_state_b, next_action_b, reward_b, reward_a, agent_a.id)


# ---------------------------------------------------------------------------
# Network reconstructor
# ---------------------------------------------------------------------------

class NetworkReconstructor:
    """Handles periodic edge add / remove logic."""

    def __init__(
        self,
        graph: nx.Graph,
        agents: list[Agent],
        positions: dict,
        max_distance: float,
        betrayal_threshold: float,
        percent_reconnection: float,
    ):
        self.graph = graph
        self.agents = agents
        self.positions = positions
        self.max_distance = max_distance
        self.betrayal_threshold = betrayal_threshold
        self.percent_reconnection = percent_reconnection

        all_nodes = list(range(len(agents)))
        self._all_pairs = [(a, b) for a in all_nodes for b in range(a + 1, len(agents))]

    def reconstruct(self, iteration: int):
        subset_size = int(len(self._all_pairs) * self.percent_reconnection)
        evaluated = random.sample(self._all_pairs, subset_size)

        for a_id, b_id in evaluated:
            dist = euclidean_distance(self.positions[a_id], self.positions[b_id])
            if dist > self.max_distance:
                continue

            agent_a = self.agents[a_id]
            agent_b = self.agents[b_id]

            if self.graph.has_edge(a_id, b_id):
                keep = agent_a.keep_connected_to_opponent(b_id, self.betrayal_threshold, iteration)
                if keep == 0:
                    self.graph.remove_edge(a_id, b_id)
            else:
                can_a = not (isinstance(agent_a, (MoodySARSAAgent, SARSAAgent)) and b_id in agent_a.betrayal_memory)
                can_b = not (isinstance(agent_b, (MoodySARSAAgent, SARSAAgent)) and a_id in agent_b.betrayal_memory)
                if can_a and can_b:
                    self.graph.add_edge(a_id, b_id)


# ---------------------------------------------------------------------------
# Forgiveness handler
# ---------------------------------------------------------------------------

class ForgivenessHandler:
    """Clears or trims betrayal memories according to the configured mode."""

    def __init__(self, agents: list[Agent], mode: str):
        self.agents = agents
        self.mode = mode

    def trigger(self):
        learning_agents = [a for a in self.agents if isinstance(a, (MoodySARSAAgent, SARSAAgent))]
        if self.mode == "WIPE":
            for a in learning_agents:
                a.betrayal_memory.clear()
        elif self.mode == "POP":
            for a in learning_agents:
                if a.betrayal_memory:
                    a.betrayal_memory.popleft()


# ---------------------------------------------------------------------------
# Colour / mood helpers
# ---------------------------------------------------------------------------

def colors_and_moods(agents: list[Agent]) -> tuple[list[str], list]:
    colors, moods = [], []
    for agent in agents:
        colors.append(agent_display_color(agent))
        moods.append(agent.mood if isinstance(agent, MoodySARSAAgent) else None)
    return colors, moods


# ---------------------------------------------------------------------------
# Main simulation orchestrator
# ---------------------------------------------------------------------------

class Simulation:
    """
    Orchestrates the full IPD simulation loop.

    Parameters
    ----------
    state   : SimulationState  – owns live color/mood data and pause control.
              Returned by SimulationApp.attach_simulation().
    graph   : nx.Graph
    config  : dict  – must include "sim_name" key (used for CSV filename)
    """

    RECONSTRUCTION_INTERVAL = 10

    def __init__(self, state, graph: nx.Graph, config: dict):
        self.state = state
        self.graph = graph
        self.config = config
        self._setup()

    def _setup(self):
        import os
        from datetime import datetime

        cfg = self.config
        n = cfg["num_nodes"]
        self.num_nodes = n
        self.num_games = cfg["num_games_per_pair"]
        self.forgiveness_mode = cfg["forgiveness_mode"]
        self.dimensions = int(math.sqrt(n))
        self.positions = cfg["node_positions"]

        # Build CSV filename: {name}-{YYYY-MM-DD_HH-MM-SS}
        raw_name = cfg.get("sim_name", "").strip()
        safe_name = raw_name if raw_name else f"simulation-{self.state.sim_id}"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_stem = f"{safe_name}-{timestamp}"
        os.makedirs("stats", exist_ok=True)

        n_states, n_actions = 101, 2
        self.agents = AgentFactory.from_assignment(
            n, cfg["agent_assignment"], n_states, n_actions
        )

        self.env = PrisonersDilemmaEnvironment(n_agents=n, n_states=n_states)
        self.game_runner = GameRunner(self.env)

        self.reconstructor = NetworkReconstructor(
            graph=self.graph,
            agents=self.agents,
            positions=self.positions,
            max_distance=cfg["max_connection_distance"],
            betrayal_threshold=cfg["average_considered_betrayal"],
            percent_reconnection=cfg["percent_reconnection"],
        )

        self.forgiveness = ForgivenessHandler(self.agents, self.forgiveness_mode)

        max_degrees = GraphFactory.max_degrees(
            self.graph, self.positions, cfg["max_connection_distance"]
        )
        self.tracker = AgentTracker(self.agents, self.dimensions, n, max_degrees)
        # Give tracker the named CSV paths
        self.tracker.csv_path_mood  = os.path.join("stats", f"{csv_stem}-mood.csv")
        self.tracker.csv_path_score = os.path.join("stats", f"{csv_stem}-score.csv")

    def run(self):
        """
        Blocking loop – run inside a daemon thread.
        Calls state.wait_if_paused() each iteration so the pause button
        blocks the thread cleanly without busy-waiting.
        """
        ri = self.RECONSTRUCTION_INTERVAL

        for i in range(self.num_games):
            self.state.wait_if_paused()

            for edge in self.graph.edges():
                self.game_runner.play(self.agents[edge[0]], self.agents[edge[1]])

            if (i + 1) % (ri * 10) == 1:
                current_degrees = dict(self.graph.degree())
                self.tracker.track_types_metrics(current_degrees)
                # Push latest series snapshot into SimulationState so the
                # Dash interval callback can read it without touching AgentTracker
                self.state.update_metrics(self.tracker.get_latest_series())

            if (i + 1) % (ri * 100) == 1:
                self.env.reset()
                self.forgiveness.trigger()
                print(f"[Sim {self.state.sim_id}] Reset at iteration {i + 1}")

            if (i + 1) % ri == 0:
                # print(f"[Sim {self.state.sim_id}] Reconstruction at iteration {i + 1}")
                self.reconstructor.reconstruct(i)
                colors, moods = colors_and_moods(self.agents)
                self.state.update(colors, moods)
                self.state.update_iteration(i + 1)
                time.sleep(0.05)

        self.state.finished = True
        print(f"[Sim {self.state.sim_id}] Finished. CSVs saved to stats/")


# ---------------------------------------------------------------------------
# Thread pool manager
# ---------------------------------------------------------------------------

class SimulationThreadPool:
    """
    Manages all running simulation threads.

    Each simulation gets its own daemon thread. The pool tracks them so
    callers can query status, cancel, or wait for completion.

    Usage
    -----
    pool = SimulationThreadPool()
    future = pool.submit(sim)          # starts sim.run() in a thread
    pool.cancel(sim_id)                # pauses + marks cancelled
    pool.status()                      # {sim_id: "running"|"paused"|"done"|"cancelled"}
    pool.shutdown(wait=False)          # stops accepting new work
    """

    def __init__(self, max_workers: int = 16):
        # ThreadPoolExecutor manages the thread lifecycle cleanly
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ipd-sim",
        )
        self._futures: dict[str, object] = {}   # sim_id → Future
        self._sims:    dict[str, "Simulation"] = {}  # sim_id → Simulation
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, sim: "Simulation") -> None:
        """Start sim.run() in a worker thread. Non-blocking."""
        sid = sim.state.sim_id
        future = self._executor.submit(sim.run)
        with self._lock:
            self._futures[sid] = future
            self._sims[sid] = sim
        future.add_done_callback(lambda f: self._on_done(sid, f))
        print(f"[Pool] Sim {sid} submitted to thread pool.")

    def pause(self, sim_id: str) -> None:
        """Pause a running simulation (thread blocks at next iteration)."""
        sim = self._get_sim(sim_id)
        if sim:
            sim.state.pause()

    def resume(self, sim_id: str) -> None:
        """Resume a paused simulation."""
        sim = self._get_sim(sim_id)
        if sim:
            sim.state.resume()

    def cancel(self, sim_id: str) -> None:
        """
        Best-effort cancellation. Pauses the sim so its thread exits
        at the next wait_if_paused() — then marks it finished.
        """
        sim = self._get_sim(sim_id)
        if not sim:
            return
        sim.state.pause()           # block the thread
        sim.state.finished = True   # signal done
        sim.state.resume()          # unblock so it can exit cleanly
        print(f"[Pool] Sim {sim_id} cancelled.")

    def status(self) -> dict[str, str]:
        """Return {sim_id: status_string} for all known sims."""
        result = {}
        with self._lock:
            items = list(self._sims.items())
        for sid, sim in items:
            if sim.state.finished:
                result[sid] = "done"
            elif sim.state.paused:
                result[sid] = "paused"
            else:
                result[sid] = "running"
        return result

    def running_count(self) -> int:
        return sum(1 for s in self.status().values() if s == "running")

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the pool. Pass wait=True to block until all finish."""
        self._executor.shutdown(wait=wait)

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_sim(self, sim_id: str) -> "Simulation | None":
        with self._lock:
            return self._sims.get(sim_id)

    def _on_done(self, sim_id: str, future) -> None:
        """Called by the executor when a thread finishes."""
        exc = future.exception()
        if exc:
            print(f"[Pool] Sim {sim_id} raised an exception: {exc}")
        else:
            print(f"[Pool] Sim {sim_id} thread completed cleanly.")


# Module-level singleton pool — imported and used by gui_app.py
_pool = SimulationThreadPool()

def get_pool() -> SimulationThreadPool:
    return _pool