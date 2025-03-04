# server.py

from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from mesa.visualization.user_param import Slider
from model import GlobalDevelopmentModel

def agent_portrayal(agent):
    """
    Return a portrayal dictionary for an agent.
    For MicroAgents (which have a 'country_id'), we display a circle
    colored by employment status. For MacroAgents, we display a rectangle.
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
        # MacroAgent (if visualized; here we mark it as gray).
        return {
            "Shape": "rect",
            "Color": "gray",
            "w": 1,
            "h": 1,
            "Layer": 1,
            "Label": "Macro",
        }

# Create a space component to visualize the network of MicroAgents.
network_component = make_space_component(agent_portrayal, canvas_width=500, canvas_height=500)

# Create plot components for global metrics.
chart_avg_innovation = make_plot_component(
    [{"Label": "Global_Avg_Innovation", "Color": "Black"}],
    title="Average Innovation"
)
chart_avg_resources = make_plot_component(
    [{"Label": "Global_Avg_Resources", "Color": "Black"}],
    title="Average Resources"
)
chart_employment_rate = make_plot_component(
    [{"Label": "Global_Employment_Rate", "Color": "Black"}],
    title="Employment Rate"
)
chart_gini = make_plot_component(
    [{"Label": "Global_Gini_Resources", "Color": "Black"}],
    title="Gini Coefficient"
)

# Define user-settable parameters using the Slider class.
model_params = {
    "num_countries": Slider("Number of Countries", 3, 1, 6, 1),
    "agents_per_country": Slider("Agents per Country", 30, 5, 100, 5),
    "innovation_factor": Slider("Innovation Factor", 0.03, 0.0, 0.1, 0.01)
}

# Combine all visualization components.
components = [network_component, chart_avg_innovation, chart_avg_resources, chart_employment_rate, chart_gini]

# Create the SolaraViz instance (the new visualization server).
viz = SolaraViz(GlobalDevelopmentModel, model_params, components, "Global Development Model")
viz.port = 8521  # Set the port for the visualization server.

if __name__ == "__main__":
    viz.launch()
