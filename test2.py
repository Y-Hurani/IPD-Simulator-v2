'''
THIS FILE IS USED TO RUN MULTIPLE SIMULATIONS AT THE SAME TIME. THIS CAN BE USED BY THE USER, HOWEVER
I KEPT IT TO BE USED BY ME TO HARVEST DATA IN BULK AND OVER THE SPAN OF MULTIPLE DAYS AT THE SAME TIME
'''
import time
from collections import deque
import numpy as np
import random
import networkx as nx
import math
from pyvis.network import Network
from Agent import Agent
from CooperativeAgent import CooperativeAgent
from TFTAgent import TFTAgent
from WSLSAgent import WSLSAgent
from DefectingAgent import DefectingAgent
from SARSAAgent import SARSAAgent
from AgentTracker import AgentTracker
from MoodySARSAAgent import MoodySARSAAgent
from visualization import create_dash_app 
from visualization import nx_to_cytoscape
import threading


class PrisonersDilemmaEnvironment:
    def __init__(self, n_agents, n_states=10):
        self.n_agents = n_agents  # Total number of agents
        self.n_states = n_states 
        self.total = [0, 0]  # Tracks total cooperation/defection counts

    def reset(self):
        self.total = [0, 0]

    def step(self, id_A, id_B, action_A, action_B):
        """
        Simulate a single game between two agents and update trust levels.
        """
        # Determine rewards based on actions
        if action_A == 1 and action_B == 1:  # Both Cooperate
            reward_A, reward_B = 3, 3
            #reward_A +=  self.trust_levels[id_A][id_B] * 0.1
            #reward_B += self.trust_levels[id_B][id_A] * 0.1
            self.total[1] += 2  # Cooperation counter
        elif action_A == 1 and action_B == 0:  # A Cooperates, B Defects
            reward_A, reward_B = 0, 5
            self.total[0] += 1  # Defection counter
        elif action_A == 0 and action_B == 1:  # A Defects, B Cooperates
            reward_A, reward_B = 5, 0
            self.total[0] += 1
        else:  # Both Defect
            reward_A, reward_B = 1, 1
            self.total[0] += 2

        return (reward_A, reward_B)

    def update_trust(self, agent_A, agent_B, action_A, action_B):
        """
        Depreciated.
        """
        if action_A == 1:  # Agent A cooperated
            self.trust_levels[agent_B][agent_A] = min(self.n_states - 1, self.trust_levels[agent_B][agent_A] + 1)
        else:  # Agent A defected
            self.trust_levels[agent_B][agent_A] = max(0, self.trust_levels[agent_B][agent_A] - 1)

        if action_B == 1:  # Agent B cooperated
            self.trust_levels[agent_A][agent_B] = min(self.n_states - 1, self.trust_levels[agent_A][agent_B] + 1)
        else:  # Agent B defected
            self.trust_levels[agent_A][agent_B] = max(0, self.trust_levels[agent_A][agent_B] - 1)

def play_game(agent_A, agent_B, env, id_A, id_B, fixed=0):
    """
    Plays one game of IPD between two agents, retrieving states from the environment.
    """

    state_A = agent_A.mood if isinstance(agent_A, MoodySARSAAgent) else 50
    state_B = agent_B.mood if isinstance(agent_B, MoodySARSAAgent) else 50

    # Agents choose actions
    action_A = agent_A.choose_action(state_B, id_B, fixed)
    action_B = agent_B.choose_action(state_A, id_A, fixed)

    # Step in the environment to get next states and rewards
    (reward_A, reward_B) = env.step(id_A, id_B, action_A, action_B)

    next_state_A = max(0, min(99, agent_A.mood + int(reward_A - agent_A.prev_omegas[agent_B.id]))) if isinstance(agent_A, MoodySARSAAgent) else 50
    next_state_B = max(0, min(99, agent_B.mood + int(reward_B - agent_B.prev_omegas[agent_A.id]))) if isinstance(agent_B, MoodySARSAAgent) else 50

    # Agents choose next actions
    next_action_A = agent_A.choose_action(next_state_A, id_B, fixed)
    next_action_B = agent_B.choose_action(next_state_B, id_A, fixed)

    # Calculate new average payoffs
    new_average_A = agent_A.average_payoff + ((reward_A - agent_A.average_payoff) / (agent_A.total_games + 1)) 
    new_average_B = agent_B.average_payoff + ((reward_B - agent_B.average_payoff) / (agent_B.total_games + 1)) 

    # Update Q-values for both agents
    agent_A.after_game_function(state_A, action_A, next_state_A, next_action_A, reward_A, reward_B, id_B)
    agent_B.after_game_function(state_B, action_B, next_state_B, next_action_B, reward_B, reward_A, id_A)


node_positions = {} # Coordinates of each node
def calculate_distance(node_a, node_b, positions=node_positions):
    """
    Calculate Euclidean distance between two nodes.
    :param node_a: First node ID
    :param node_b: Second node ID
    :param positions: Dictionary of node positions
    :return: Distance
    """
    x1, y1 = positions[node_a]
    x2, y2 = positions[node_b]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_networkx_graph(num_nodes=25, num_edges=40):
    graph = nx.Graph()
    # Add nodes
    graph.add_nodes_from(range(num_nodes))  # Create nodes with ID's from 0 to num_nodes -1
    return graph

def agent_class_condition(node, agents):
    """
    Determines the color of a node based on the agent's class.
    :param node: The node ID.
    :param agents: A dictionary of agents where keys are node IDs and values are agent instances.
    :return: A string representing the node color.
    """
    if isinstance(agents[node], MoodySARSAAgent):
        return 'green'
    elif isinstance(agents[node], CooperativeAgent):
        return 'blue'
    else:
        return 'orange'  # Default color for other classes

def add_edges_to_graph(graph):
    # Randomly create edges between nodes
    while graph.number_of_edges() < num_edges:
        node_a = random.randint(0, num_nodes - 1)
        node_b = random.randint(0, num_nodes - 1)

        # No self-loops or duplicates
        distance = calculate_distance(node_a, node_b)
        if node_a != node_b and not graph.has_edge(node_a, node_b) and distance < max_connection_distance:
            graph.add_edge(node_a, node_b)

        return graph

def visualize(graph, grid_size=7):
    net = Network(notebook=True, directed=False, cdn_resources='remote')  # Non-directed graph

    # Define fixed positions for the nodes in a grid
    for node in graph.nodes:
        row = node // grid_size
        col = node % grid_size
        x = col * 100
        y = row * 100
        node_positions[node] = (x, y)

    for node, (x, y) in node_positions.items():
        net.add_node(node, x=x, y=y, fixed=True)

    for edge in graph.edges:
        net.add_edge(*edge)

    return net

def generate_agents(n_agents, weights, n_states, n_actions):
    """
    Generate a list of agents based on weighted probabilities.

    :param n_agents: Total number of agents to generate.
    :param weights: A dictionary containing the weights for each agent type.
                    Example: {"MoodySARSAAgent": 0.5, "CooperativeAgent": 0.2, "DefectingAgent": 0.2, "TFTAgent": 0.1}
    :param n_states: Number of states for the agents.
    :param n_actions: Number of actions for the agents.
    :return: A list of generated agents.
    """
    agent_classes = {
        "MoodySARSAAgent": MoodySARSAAgent,
        "SARSAAgent": SARSAAgent,
        "CooperativeAgent": CooperativeAgent,
        "DefectingAgent": DefectingAgent,
        "TFTAgent": TFTAgent,
        "WSLSAgent": WSLSAgent
    }

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {key: val / total_weight for key, val in weights.items()}

    # list of agent types and probabilities
    agent_types = list(agent_classes.keys())
    probabilities = [normalized_weights[agent_type] for agent_type in agent_types]

    # Generate agents based on weight
    agents = []
    for agent_id in range(n_agents):
        chosen_type = random.choices(agent_types, probabilities, k=1)[0]
        agent_class = agent_classes[chosen_type]
        agents.append(agent_class(id=agent_id, n_states=n_states, n_actions=n_actions, n_agents=n_agents))

    return agents

def mood_to_color(mood):
    """
    Converts mood (1 to 100) to a grayscale hex color.
    """
    # Clamp to range
    mood = max(1, min(100, mood))
    # Map mood to grayscale intensity (0 = black, 255 = white)
    intensity = int((mood / 100) * 255)
    hex_value = f'{intensity:02x}'  # 2-digit hex
    return f'#{hex_value}{hex_value}{hex_value}'

def update_colors_moods():
    colors, moods = [], []
    for agent in agents:
        moods.append('None')
        if isinstance(agent, CooperativeAgent):
            colors.append('green')
        elif isinstance(agent, MoodySARSAAgent):
            colors.append(mood_to_color(agent.mood))
            moods[-1] = agent.mood
        elif isinstance(agent, SARSAAgent):
            colors.append('yellow')
        elif isinstance(agent, TFTAgent):
            colors.append('pink')
        elif isinstance(agent, DefectingAgent):
            colors.append('red')
        else:
            colors.append('orange')
    return colors, moods

def trigger_forgiveness(mode):
    if mode == 'WIPE':
        for agent in agents:
            if isinstance(agent, MoodySARSAAgent) or isinstance(agent, SARSAAgent):
                agent.betrayal_memory.clear()
    elif mode == 'POP':
        for agent in agents:
            if isinstance(agent, MoodySARSAAgent) or isinstance(agent, SARSAAgent):
                if len(agent.betrayal_memory) > 0:
                    agent.betrayal_memory.popleft()
    else:
        print('No applicable mode')

def calculate_max_degrees(graph, max_connection_distance, positions=node_positions):
    """
    Calculate maximum possible degree for each node based on distance constraints.
    
    Args:
        graph: NetworkX graph object
        max_connection_distance: Maximum allowed distance between connected nodes
        positions: Dictionary mapping node IDs to their (x, y) coordinates
        
    Returns:
        Dictionary mapping each node ID to its maximum possible degree
    """
    max_degrees = {}
    
    # For each node in the graph
    for node_a in graph.nodes():
        # Count how many other nodes are within the max connection distance
        possible_connections = 0
        
        for node_b in graph.nodes():
            # Skip self
            if node_a == node_b:
                continue
                
            # Calculate distance between nodes
            distance = calculate_distance(node_a, node_b, positions)
            
            # If distance is within limit, this is a possible connection
            if distance < max_connection_distance:
                possible_connections += 1
                
        # Store the maximum possible degree for this node
        max_degrees[node_a] = possible_connections
        
    return max_degrees


# Create a graph and visualize it
num_nodes = 100
num_edges = 50
dimensions = int(math.sqrt(num_nodes))
max_connection_distance = 250
graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
# Initiate variables
n_agents = num_nodes
n_states = 101
n_actions = 2  # Actions: 0 = defect, 1 = coop
fixed = 0
weights = {
    "SARSAAgent": 0.00,
    "MoodySARSAAgent": 1.00,
    "CooperativeAgent": 0.00,  
    "DefectingAgent": 0.00,  
    "TFTAgent": 0.0,
    "WSLSAgent": 0.0
}

agents = generate_agents(n_agents, weights, n_states, n_actions)
colors, moods = update_colors_moods()
visualize(graph, dimensions)
add_edges_to_graph(graph)
env = PrisonersDilemmaEnvironment(n_agents=n_agents, n_states=n_states)


# pair matches
num_games_per_pair = 50000
removed_edges_list = []
reconstruction_interval = 10  # number of rounds before reconstruction
percent_reconnection = 0.20  # % of the population for random reconnection
average_considered_betrayal = 2.5


# Start the Dash app in a separate thread or process
app = create_dash_app(graph, colors, dimensions)
thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False))
thread.daemon = True
thread.start()
possible_pairs = [(a, b) for a in range(n_agents) for b in range(a + 1, n_agents)]
max_degrees = calculate_max_degrees(graph, max_connection_distance)
print(max_degrees)
subset_size = int(len(possible_pairs) * percent_reconnection)

# Initialize tracking class AgentTracker
tracker = AgentTracker(agents, dimensions, 100, max_degrees)


    # game loop to update the Dash graph
for limit in [250]:#, 250000]:
    for percentage_spread in range(0, 1):
        sarsa_spread = percentage_spread * 0.10
        weights['SARSAAgent'] = sarsa_spread
        weights['MoodySARSAAgent'] = 1 - sarsa_spread
        max_connection_distance = limit
        agents = generate_agents(n_agents, weights, n_states, n_actions)
        tracker = AgentTracker(agents, dimensions, 100, max_degrees)
        tracker.csv_path_mood = f"stats/agent_mood_by_layer_{percentage_spread}_{limit}.csv"
        tracker.csv_path_score = f"stats/agent_mood_by_layer_{percentage_spread}_{limit}csv"
        for i in range(num_games_per_pair):
            for edge in graph.edges():
                play_game(agents[edge[0]], agents[edge[1]], env, edge[0], edge[1], fixed)
            if fixed >= 1:
                fixed = fixed - 1

            if (i + 1) % (reconstruction_interval * 10) == 1:
                current_degrees = {node: graph.degree(node) for node in graph.nodes()}
                tracker.track_types_metrics(current_degrees)
                trigger_forgiveness('POP')
                
            if (i + 1) % (reconstruction_interval * 100) == 1:
                print(env.total)
                env.reset()
                
                
            if (i + 1) % reconstruction_interval == 0:
                print(f"Reconstruction event at iteration {i + 1}")

                evaluated_pairs = random.sample(possible_pairs, subset_size)

                for agent_a, agent_b in evaluated_pairs:
                    distance = calculate_distance(agent_b, agent_a)
                    if distance > max_connection_distance:
                        continue
                    if graph.has_edge(agent_a, agent_b):
                        decision_a = agents[agent_a].keep_connected_to_opponent(agent_b, average_considered_betrayal, i)
                        decision_b = agents[agent_b].keep_connected_to_opponent(agent_a, average_considered_betrayal, i)
                        if decision_a == 0:# or decision_b == 0:
                            graph.remove_edge(agent_a, agent_b)
                    else:
                        # If either of the agents are moody and hold grudge against opponent, dont connect
                        condition_A, condition_B = True, True
                        if isinstance(agents[agent_a], MoodySARSAAgent) or isinstance(agents[agent_a], SARSAAgent):
                            if agent_b in agents[agent_a].betrayal_memory:
                                condition_A = False

                        if isinstance(agents[agent_b], MoodySARSAAgent) or isinstance(agents[agent_b], SARSAAgent):
                            if agent_a in agents[agent_b].betrayal_memory:
                                condition_B = False

                        if condition_A and condition_B:
                            graph.add_edge(agent_a, agent_b)

                # Trigger Dash Cytoscape to redraw the updated graph
                # This replaces `network_visualization.show("network.html")`
                colors, moods = update_colors_moods()
                elements = nx_to_cytoscape(graph, colors, dimensions, moods)
                app.layout.children[-1].elements = elements
                app.update_data(colors, moods)
                #for agent in agents: # Epsilon decay (Experimental)
                #    if isinstance(agent, MoodySARSAAgent):
                #        agent.set_epsilon(agent.epsilon * 0.9995)
                time.sleep(0.05)