import math
import random
import numpy as np
from Agent import Agent
from DefectingAgent import DefectingAgent # in case to implement impossibility to disconnect from control agents
from collections import deque


class MoodySARSAAgent(Agent):
    def __init__(self, n_states, n_actions, n_agents, id, alpha=0.1, gamma=0.95, epsilon=0.1):
        super().__init__(id, n_states, n_actions, n_agents)
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.epsilon = epsilon  # exploration
        self.betrayal_memory = deque()
        self.mood = 50 # Mood value (1 to 100, neutral mood = 50)
        self.prev_omegas = {i: 0 for i in range(n_agents)}

        # Q-tables for each opponent agent
        # dictionary with key = agent id and the value is the q table
        self.q_tables = {i: np.zeros((n_states, n_actions))
                         for i in range(n_agents)}


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def after_game_function(self, state, action, next_state, next_action, reward, opponent_reward, opponent_id):
        self.update_q_value(state, action, reward, opponent_id)
        self.update_mood(opponent_id, reward, opponent_reward)
        self.update_memory(opponent_id, reward)
        self.update_average_payoff(reward)
    
    def calculate_new_omega(self, opponent_id, reward, opponent_reward):
        # Calculate average rewards including the new payoff
        self_payoffs = self.memories[opponent_id]
        opponent_payoffs = []
        for rewarded in self_payoffs:
            match rewarded:
                case 0:
                    opponent_payoffs.append(5)
                    break
                case 1:
                    opponent_payoffs.append(1)
                    break
                case 3:
                    opponent_payoffs.append(3)
                    break
                case 5:
                    opponent_payoffs.append(0)
                    break

        avg_self_reward_t = np.mean(self_payoffs[-19:] + [reward]) if self_payoffs else reward

        avg_opponent_reward_t = np.mean(opponent_payoffs[-19:] + [opponent_reward]) if opponent_payoffs else opponent_reward

        # Calculate alpha and omega (Homo Egualis adjustment)
        alpha = (100 - self.mood) / 100
        beta = alpha
        omega = avg_self_reward_t - (alpha * max(avg_opponent_reward_t - avg_self_reward_t, 0)) - (beta * max(avg_self_reward_t - avg_opponent_reward_t, 0))
        return omega

    def update_mood(self, opponent_id, reward, opponent_reward):
        """
        Adjust mood based on performance and fairness using opponent payoff history
        """
        omega = self.calculate_new_omega(opponent_id, reward, opponent_reward)
        self.mood += int(reward - self.prev_omegas[opponent_id])
        self.mood = max(0, min(100, self.mood))  # Clamp mood to [1, 100]
        self.prev_omegas[opponent_id] = omega

    def compute_mood_adjusted_estimate(self, opponent_id):
        """
        Calculate mood-adjusted estimate
        """
        memory = self.memories[opponent_id]
        if not memory:
            return 0

        mood_factor = (100 - self.mood) / 100  # Scales between 0 and 1
        max_depth = 20  # Set the maximum depth of memory to consider
        depth = math.ceil(mood_factor * max_depth)  # Scales depth based on mood
        relevant_memory = memory[-depth:] if depth > 0 else memory

        return np.mean(relevant_memory) if relevant_memory else 0

    def choose_action(self, state, opponent_id, fixed=0):
        q_table = self.q_tables[opponent_id]
        decision_epsilon = self.epsilon
        if fixed:
            return 1

        # Exploit
        max_value = np.max(q_table[state, :])
        max_actions = [action for action, value in enumerate(q_table[state, :]) if
                       value == max_value]  # Choose action using epsilon-greedy policy for the specific opponent.
        chosen_action = random.choice(max_actions)  # Randomly choose among the actions with max Q-value

        if self.mood < 30 and chosen_action == 1:
            decision_epsilon = 0.9
        if self.mood > 70 and chosen_action == 0:
            decision_epsilon = 0.9

        if random.uniform(0, 1) < decision_epsilon:  # Random chance based on epsilon to choose randomly instead
            if self.mood > 90:
                chosen_action = random.choices(population=range(self.n_actions), weights=[0.5, 0.5])[0]
            else:
                chosen_action = random.choices(population=range(self.n_actions), weights=[0.5, 0.5])[0]
        return chosen_action

    def update_q_value(self, state, action, reward, opponent_id):
        """
        Modify SARSA Q-value update rule to mood-adjusted
        """
        q_table = self.q_tables[opponent_id]
        mood_adjusted_estimate = self.compute_mood_adjusted_estimate(opponent_id)

        # Standard SARSA update with mood adjusted
        td_target = reward + self.gamma * mood_adjusted_estimate
        td_error = td_target - q_table[state, action]
        q_table[state, action] += self.alpha * td_error

    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def average_reward(self, opponent_id, cap=20):
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id][:cap])

    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
        # DefectingAgent stay connected
        # if opponent_id in range(110, 120):
        #    return 1

        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:
            self.betrayal_memory.append(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            return 1
