import numpy as np
class Agent:
    def __init__(self, id, n_states, n_actions, n_agents):

        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.average_payoff = 0
        self.total_games = 1   # For stats tracking
        self.memories = {opponent_id: [] for opponent_id in range(n_agents)}  # Track memories for each opponent
        self.total_cooperation = 0  # For stats tracking
        self.betrayal_memory = set() # Holding grudge memory
        self.mood = 50


    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def update_average_payoff(self, payoff):
        self.total_games += 1
        self.average_payoff += (payoff - self.average_payoff) / self.total_games
        if payoff == 3 or payoff == 0:
            self.total_cooperation += 1
        
    def choose_action(self, state, opponent_id, **kwargs):
        """
        Select an action given the current state and opponent.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by a subclass")

    def after_game_function(self, state, action, reward, opponent_reward, opponent_id, opponent_average, **kwargs):
        """
        Perform updates after each game round (Q-value updates, mood updates, memory tracking).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by a subclass")

    def reset(self):
        """
        Reset agent-specific attributes (if needed) between episodes or games.
        Optional implementation in subclasses.
        """
        pass

    def average_reward(self, opponent_id):  # For stats tracking
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id])


    def keep_connected_to_opponent(self, opponent_id):
        """
        Decide whether to stay connected to an opponent or to cut ties. Must be implemented by subclass.
        """
        return 1