import numpy as np
from Agent import Agent


class WSLSAgent(Agent):
    def __init__(self, id, n_states, n_actions, n_agents):
        super().__init__(id, n_states, n_actions, n_agents)
        self.last_opponent_actions = {i: 1 for i in range(n_agents)}  # Default to cooperation (action=1)

    def choose_action(self, state, opponent_id, *args):
        """
        WSLS logic: Stick with winning move, otherwise switch.
        """
        return self.last_opponent_actions[opponent_id]

    def after_game_function(self, state, action, reward, opponent_reward, opponent_id, opponent_average, *args):
        """
        Update the agent's behavior after a game round.
        """
        # Track opponent's last action based on received rewards.
        if opponent_reward in {5, 1}:  # Defection rewards
            current_value = self.last_opponent_actions[opponent_id]
            self.last_opponent_actions[opponent_id] = 1 - current_value # Flip the Value

        self.update_memory(opponent_id, reward)

        # Update average payoff
        self.total_games += 1
        self.average_payoff = ((self.average_payoff * (self.total_games - 1)) + reward) / self.total_games

    def reset(self):
        """
        Reset TFT agent's state for a new episode.
        """
        self.last_opponent_action = 1  # Reset to cooperation

    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:

            self.betrayal_memory.add(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            return 1

