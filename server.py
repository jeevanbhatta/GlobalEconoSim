# server.py

from mesa.visualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule, NetworkModule
from model import GlobalDevelopmentModel

def network_portrayal(G):
    """
    Provide a portrayal (visual representation) of the network and its agents.
    G is a networkx Graph where each node is an agent's unique_id.
    """
    portrayal = dict()
    nodes = []
    edges = []
    
    for node_id in G.nodes():
        agent = G.nodes[node_id]["agent"][0]  # Mesa typically stores agents in a list
        if agent is not None:
            if hasattr(agent, "country_id"):
                # It's a MicroAgent
                portrayal_node = {
                    "id": node_id,
                    "size": 5,  # smaller for micro-agents
                    "shape": "circle",
                    "label": f"{int(agent.resources)}",
                    # You might color by the agent's country or employment
                    "color": "blue" if agent.employment == "employed" else "red",
                }
            else:
                # Possibly a MacroAgent (not typically visualized as nodes in this example)
                portrayal_node = {
                    "id": node_id,
                    "size": 10,
                    "shape": "rect",
                    "label": "Macro",
                }
            nodes.append(portrayal_node)

    for (source, target) in G.edges():
        edges.append({"source": source, "target": target})

    portrayal["nodes"] = nodes
    portrayal["edges"] = edges
    return portrayal


# Create a network module that uses our portrayal
network = NetworkModule(network_portrayal, 500, 500, library="d3")

# Create chart modules for global metrics
chart_avg_innovation = ChartModule(
    [{"Label": "Global_Avg_Innovation", "Color": "Black"}],
    data_collector_name='datacollector'
)
chart_avg_resources = ChartModule(
    [{"Label": "Global_Avg_Resources", "Color": "Black"}],
    data_collector_name='datacollector'
)
chart_employment_rate = ChartModule(
    [{"Label": "Global_Employment_Rate", "Color": "Black"}],
    data_collector_name='datacollector'
)
chart_gini = ChartModule(
    [{"Label": "Global_Gini_Resources", "Color": "Black"}],
    data_collector_name='datacollector'
)

# Example: user-settable parameters
model_params = {
    "num_countries": UserSettableParameter("slider", "Number of Countries", 3, 1, 6, 1),
    "agents_per_country": UserSettableParameter("slider", "Agents per Country", 30, 5, 100, 5),
    "innovation_factor": UserSettableParameter("slider", "Innovation Factor", 0.03, 0.0, 0.1, 0.01)
}

# Construct the server
server = ModularServer(
    GlobalDevelopmentModel,
    [network, chart_avg_innovation, chart_avg_resources, chart_employment_rate, chart_gini],
    "Global Development Model",
    model_params
)
server.port = 8521  # default Mesa port

if __name__ == "__main__":
    server.launch()
