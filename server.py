from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from model import GlobalDevelopmentModel


def agent_portrayal(agent):
    """
    Return a portrayal dictionary for an agent.
    For MicroAgents (which have a 'country_id'), display a circle
    colored by employment status; for MacroAgents, display a rectangle.
    """
    if hasattr(agent, "country_id"):
        # MicroAgent: Display as a circle, blue if employed, red otherwise.
        return {
            "Shape": "circle",
            "Color": "blue" if agent.employment == "employed" else "red",
            "r": 0.5,
            "Layer": 0,
            "Label": str(int(agent.resources)),
        }
    else:
        # MacroAgent: Display as a gray rectangle (if visible on the network).
        return {
            "Shape": "rect",
            "Color": "gray",
            "w": 1,
            "h": 1,
            "Layer": 1,
            "Label": "Macro",
        }


# 1. Create a NetworkModule to visualize the network of agents.
#    - You can specify width, height, and optional library="d3" (the default).
network_element = NetworkModule(agent_portrayal, 500, 500, library="d3")

# 2. Create ChartModules to plot the time evolution of key metrics from DataCollector.
#    - Use data_collector_name='datacollector' to match the model’s DataCollector name.
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


# 3. Define user-settable parameters (sliders). 
#    This lets users modify model parameters at runtime in the browser.
model_params = {
    "num_countries": UserSettableParameter(
        "slider", 
        "Number of Countries",
        value=3,       # default
        min_value=1, 
        max_value=6, 
        step=1
    ),
    "agents_per_country": UserSettableParameter(
        "slider",
        "Agents per Country",
        value=30,      # default
        min_value=5,
        max_value=100,
        step=5
    ),
    "innovation_factor": UserSettableParameter(
        "slider",
        "Innovation Factor",
        value=0.03,     # default
        min_value=0.0,
        max_value=0.1,
        step=0.01
    ),
}

# 4. Create the ModularServer, providing:
#    - The model class
#    - A list of visualization modules
#    - A name for the interface window
#    - The model_params dict so the user can adjust them in the UI
server = ModularServer(
    GlobalDevelopmentModel,
    [
        network_element,
        chart_avg_innovation,
        chart_avg_resources,
        chart_employment_rate,
        chart_gini
    ],
    "Global Development Model",
    model_params
)

# 5. Choose a port for the server to run on (default is 8521).
server.port = 8521

# 6. Launch the web server for interactive simulation.
if __name__ == "__main__":
    server.launch()
