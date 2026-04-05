import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from MoodySARSAAgent import MoodySARSAAgent
from SARSAAgent import SARSAAgent
from TFTAgent import TFTAgent
from WSLSAgent import WSLSAgent


# Classes excluded from per-class plots (deterministic controls)
_CONTROL_CLASSES = ()   # CooperativeAgent / DefectingAgent excluded at call site


class AgentTracker:
    def __init__(self, agents, layer_size, tracking_frequency, max_degrees,
                 clear_existing=True):
        self.agents = agents
        self.layer_size = layer_size
        self.max_degrees = max_degrees
        self.tracking_frequency = tracking_frequency
        self.num_layers = len(agents) // layer_size
        self.current_round = 0

        self.tracking_data = defaultdict(lambda: defaultdict(dict))

        # ── In-memory series buffers (appended each snapshot) ──────────────
        # Structure: {metric_key: [value_at_t0, value_at_t1, ...]}
        # Round numbers stored separately for the x-axis.
        self._rounds: list[int] = []
        self._series: dict[str, list[float]] = defaultdict(list)

        # CSV paths (overridden by Simulation._setup if a name is supplied)
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y %H-%M-%S")
        self.csv_path_mood  = f"stats/results_mood_{timestamp}.csv"
        self.csv_path_score = f"stats/results_score_{timestamp}.csv"
        if not os.path.exists('stats'):
            os.makedirs('stats')

        if clear_existing and (os.path.exists(self.csv_path_score) or
                                os.path.exists(self.csv_path_mood)):
            os.remove(self.csv_path_mood)  if os.path.exists(self.csv_path_mood)  else None
            os.remove(self.csv_path_score) if os.path.exists(self.csv_path_score) else None
            self.csv_exists = False
        else:
            self.csv_exists = os.path.exists(self.csv_path_score)

    # ── Public: series accessor ────────────────────────────────────────────

    def get_latest_series(self) -> dict:
        """
        Return a snapshot of all accumulated time-series data, shaped for
        direct use by Plotly callbacks — no reshaping needed on the Dash side.

        Returns
        -------
        {
          "rounds": [10, 20, 30, ...],          # x-axis for all charts

          # Per-class average payoff (excludes CooperativeAgent/DefectingAgent)
          "score": {
              "MoodySARSA": [...],
              "SARSA":      [...],
              "TFT":        [...],
              "WSLS":       [...],
          },

          # Per-class normalised degree (0–1)
          "connectivity": { same keys },

          # Per-class cooperation rate (0–1)
          "cooperation": { same keys },

          # MoodySARSA-only average mood (0–100)
          "mood": { "MoodySARSA": [...] },
        }
        """
        s = self._series
        rounds = list(self._rounds)
        return {
            "rounds": rounds,
            "score": {
                "MoodySARSA": list(s["moody_avg_score"]),
                "SARSA":      list(s["sarsa_avg_score"]),
                "TFT":        list(s["tft_avg_score"]),
                "WSLS":       list(s["wsls_avg_score"]),
            },
            "connectivity": {
                "MoodySARSA": list(s["moody_avg_connectivity"]),
                "SARSA":      list(s["sarsa_avg_connectivity"]),
                "TFT":        list(s["tft_avg_connectivity"]),
                "WSLS":       list(s["wsls_avg_connectivity"]),
            },
            "cooperation": {
                "MoodySARSA": list(s["moody_avg_cooperation"]),
                "SARSA":      list(s["sarsa_avg_cooperation"]),
                "TFT":        list(s["tft_avg_cooperation"]),
                "WSLS":       list(s["wsls_avg_cooperation"]),
            },
            "mood": {
                "MoodySARSA": list(s["moody_avg_mood"]),
            },
        }

    # ── Existing methods (unchanged except _append_series call) ───────────

    def _calculate_normalized_degree(self, agent_id, current_degrees):
        if self.max_degrees is None or current_degrees is None:
            return 0.0
        current_degree = current_degrees.get(agent_id, 0)
        max_degree     = self.max_degrees.get(agent_id, 1)
        return round(current_degree / max_degree if max_degree > 0 else 0, 3)

    def track_layer_metrics(self, current_degrees):
        """Record average mood/score per layer (legacy method, unchanged)."""
        self.current_round += self.tracking_frequency
        round_data = ({}, {}, {})

        for layer_idx in range(self.num_layers):
            start_idx    = layer_idx * self.layer_size
            end_idx      = start_idx + self.layer_size
            layer_agents = self.agents[start_idx:end_idx]

            moody_agents = [a for a in layer_agents if isinstance(a, MoodySARSAAgent)]
            avg_mood     = round(np.mean([a.mood for a in moody_agents]), 3) if moody_agents else 0
            avg_score    = round(np.mean([a.average_payoff for a in layer_agents]), 3)
            avg_conn     = round(np.mean([self._calculate_normalized_degree(a.id, current_degrees)
                                          for a in layer_agents]), 3)

            self.tracking_data[self.current_round][layer_idx] = {
                'avg_mood': avg_mood, 'avg_score': avg_score,
                'avg_connectivity': avg_conn,
            }
            round_data[0][f'layer_{layer_idx}_mood']         = avg_mood
            round_data[1][f'layer_{layer_idx}_score']        = avg_score
            round_data[2][f'layer_{layer_idx}_connectivity'] = avg_conn

        round_data[0]['round'] = round_data[1]['round'] = round_data[2]['round'] = self.current_round
        self._write_to_csv(round_data, mode='layer')

    def track_types_metrics(self, current_degrees):
        """
        Record per-class metrics and append to in-memory series buffers.
        Called by Simulation.run() every ri*10 iterations.
        """
        def _metrics(agents_list, include_mood=False):
            if not agents_list:
                base = {"score": 0, "connectivity": 0, "cooperation": 0}
                if include_mood:
                    base["mood"] = 0
                return base
            # Guard against division-by-zero on total_games
            coop_rates = [
                a.total_cooperation / a.total_games
                for a in agents_list if a.total_games > 0
            ]
            result = {
                "score":        round(np.mean([a.average_payoff for a in agents_list]), 3),
                "connectivity": round(np.mean([self._calculate_normalized_degree(a.id, current_degrees)
                                               for a in agents_list]), 3),
                "cooperation":  round(np.mean(coop_rates), 3) if coop_rates else 0,
            }
            if include_mood:
                result["mood"] = round(np.mean([a.mood for a in agents_list]), 3)
            return result

        moody = _metrics([a for a in self.agents if isinstance(a, MoodySARSAAgent)], include_mood=True)
        sarsa = _metrics([a for a in self.agents if isinstance(a, SARSAAgent)])
        tft   = _metrics([a for a in self.agents if isinstance(a, TFTAgent)])
        wsls  = _metrics([a for a in self.agents if isinstance(a, WSLSAgent)])

        # ── Append to in-memory series ─────────────────────────────────────
        self._rounds.append(self.current_round)
        self._series["moody_avg_mood"].append(moody.get("mood", 0))
        self._series["moody_avg_score"].append(moody["score"])
        self._series["moody_avg_connectivity"].append(moody["connectivity"])
        self._series["moody_avg_cooperation"].append(moody["cooperation"])
        self._series["sarsa_avg_score"].append(sarsa["score"])
        self._series["sarsa_avg_connectivity"].append(sarsa["connectivity"])
        self._series["sarsa_avg_cooperation"].append(sarsa["cooperation"])
        self._series["tft_avg_score"].append(tft["score"])
        self._series["tft_avg_connectivity"].append(tft["connectivity"])
        self._series["tft_avg_cooperation"].append(tft["cooperation"])
        self._series["wsls_avg_score"].append(wsls["score"])
        self._series["wsls_avg_connectivity"].append(wsls["connectivity"])
        self._series["wsls_avg_cooperation"].append(wsls["cooperation"])

        # ── Store in tracking_data dict (legacy) ───────────────────────────
        self.tracking_data[self.current_round]['agent_types'] = {
            'moody_avg_mood':          moody.get("mood", 0),
            'moody_avg_score':         moody["score"],
            'moody_avg_connectivity':  moody["connectivity"],
            'moody_avg_cooperation':   moody["cooperation"],
            'sarsa_avg_score':         sarsa["score"],
            'sarsa_avg_connectivity':  sarsa["connectivity"],
            'sarsa_avg_cooperation':   sarsa["cooperation"],
            'tft_avg_score':           tft["score"],
            'tft_avg_connectivity':    tft["connectivity"],
            'tft_avg_cooperation':     tft["cooperation"],
            'wsls_avg_score':          wsls["score"],
            'wsls_avg_connectivity':   wsls["connectivity"],
            'wsls_avg_cooperation':    wsls["cooperation"],
        }

        # ── CSV output ─────────────────────────────────────────────────────
        csv_row = {
            'round':                   self.current_round,
            'moody_avg_mood':          moody.get("mood", 0),
            'moody_avg_score':         moody["score"],
            'moody_avg_connectivity':  moody["connectivity"],
            'moody_avg_cooperation':   moody["cooperation"],
            'sarsa_avg_score':         sarsa["score"],
            'sarsa_avg_connectivity':  sarsa["connectivity"],
            'sarsa_avg_cooperation':   sarsa["cooperation"],
            'tft_avg_score':           tft["score"],
            'tft_avg_connectivity':    tft["connectivity"],
            'tft_avg_cooperation':     tft["cooperation"],
            'wsls_avg_score':          wsls["score"],
            'wsls_avg_connectivity':   wsls["connectivity"],
            'wsls_avg_cooperation':    wsls["cooperation"],
        }
        self._write_to_csv(csv_row, mode='types')
        self.current_round += self.tracking_frequency

    def _write_to_csv(self, data, mode='layer'):
        if mode == 'layer':
            df_mood  = pd.DataFrame([data[0]])
            df_score = pd.DataFrame([data[1]])
            if not self.csv_exists:
                df_mood.to_csv(self.csv_path_mood,   index=False)
                df_score.to_csv(self.csv_path_score, index=False)
                self.csv_exists = True
            else:
                df_mood.to_csv(self.csv_path_mood,   mode='a', header=False, index=False)
                df_score.to_csv(self.csv_path_score, mode='a', header=False, index=False)
        elif mode == 'types':
            df = pd.DataFrame([data])
            if not os.path.exists(self.csv_path_score):
                df.to_csv(self.csv_path_score, index=False)
            else:
                df.to_csv(self.csv_path_score, mode='a', header=False, index=False)

    def get_summary(self):
        return dict(self.tracking_data)