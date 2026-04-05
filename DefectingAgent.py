from Agent import Agent
class DefectingAgent(Agent):
    def __init__(self, id, n_states, n_actions, n_agents):
        super().__init__(id, n_states, n_actions, n_agents)

    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def choose_action(self, state, opponent_id, whatever, **kwargs):
        """
        Always choose action 0
        """
        return 0

    def after_game_function(self, reward, opponent_id, *args):
        self.update_memory(opponent_id, reward)

    def reset(self):
        self.memories = {opponent_id: [] for opponent_id in range(self.n_agents)}

    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
            return 1 # always stays connected to exploit (evil guy)