import numpy as np
import random
from Agent import Agent
from collections import deque

class SARSAAgent(Agent):
    def __init__(self, n_states, n_actions, n_agents, id, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(id, n_states, n_actions, n_agents)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-tables for each opponent agent
        self.q_tables = {i: np.zeros((n_states, n_actions)) for i in range(n_agents)}
        self.betrayal_memory = deque() 

    def set_epsilon(self, epsilon):
        """Update the agent's exploration rate."""
        self.epsilon = epsilon

    def choose_action(self, state, opponent_id, fixed=0):
        """Choose action using epsilon-greedy strategy."""
        q_table = self.q_tables[opponent_id]

        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))  # Explore
        else:
            return np.argmax(q_table[state, :])  # Exploit

    def update_q_value(self, state, action, reward, next_state, next_action, opponent_id):
        """Perform SARSA Q-value update based on next action."""
        q_table = self.q_tables[opponent_id]
        
        # Compute the TD target
        td_target = reward + self.gamma * q_table[next_state, next_action]
        td_error = td_target - q_table[state, action]

        # Update Q-value
        q_table[state, action] += self.alpha * td_error

    def average_reward(self, opponent_id, cap=20):
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id][:cap])
    
    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:
            self.betrayal_memory.append(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            return 1

    def after_game_function(self, state, action, next_state, next_action, reward, opponent_reward, opponent_id):
        self.update_q_value(state, action, reward, next_state, next_action, opponent_id)
        self.update_average_payoff(reward)    
