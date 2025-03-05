"""
Run via:
    streamlit run server.py

Requirements:
    pip install streamlit pyvis matplotlib
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pyvis.network import Network

from model import GlobalDevelopmentModel


# ------------------------------
# HELPER: Build a PyVis network from the model's Graph
# ------------------------------
def build_pyvis_network(model):
    """
    Creates a PyVis Network from the model's networkx Graph (model.G).
    Colors micro-agents by employment status, macro-agents in gray.
    Returns the network object (which we'll convert to HTML).
    """
    net = Network(height="600px", width="100%", notebook=False, directed=False)
    net.force_atlas_2based()

    # Add nodes
    for node_id in model.G.nodes:
        agent = model.schedule._agents[node_id]
        
        if hasattr(agent, "country_id"):
            # MicroAgent
            color = "blue" if agent.employment == "employed" else "red"
            label = f"Res:{int(agent.resources)}"
            net.add_node(
                node_id,
                label=label,
                color=color,
                shape="circle",
                size=8
            )
        else:
            # MacroAgent
            net.add_node(
                node_id,
                label="Macro",
                color="gray",
                shape="square",
                size=15
            )

    # Add edges
    for source, target in model.G.edges:
        net.add_edge(source, target)

    return net


# ------------------------------
# STREAMLIT SIDEBAR PARAMETERS
# ------------------------------
st.sidebar.title("Global Development Model Parameters")
num_countries = st.sidebar.slider("Number of Countries", 1, 6, 3, 1)
agents_per_country = st.sidebar.slider("Agents per Country", 5, 100, 30, 5)
innovation_factor = st.sidebar.slider("Innovation Factor", 0.0, 0.1, 0.03, 0.01)
steps_to_run = st.sidebar.number_input("Steps to Run", min_value=1, max_value=200, value=50)
run_button = st.sidebar.button("Run / Re-run Model")


# ------------------------------
# SESSION STATE
# ------------------------------
# We store model and results in Streamlit's session state so they persist.
if "gd_model" not in st.session_state:
    st.session_state["gd_model"] = None
if "model_report_df" not in st.session_state:
    st.session_state["model_report_df"] = pd.DataFrame()
if "agent_report_df" not in st.session_state:
    st.session_state["agent_report_df"] = pd.DataFrame()
if "network_html" not in st.session_state:
    st.session_state["network_html"] = ""


def run_model():
    """
    Create and run a new GlobalDevelopmentModel with the user-specified parameters.
    Then store results (model-level DF, agent-level DF, and network HTML) in session state.
    """
    model = GlobalDevelopmentModel(
        num_countries=num_countries,
        agents_per_country=agents_per_country,
        innovation_factor=innovation_factor
    )
    
    # Step the model for the chosen number of steps
    for _ in range(int(steps_to_run)):
        model.step()

    # -- Collect data from the model
    model_report = model.datacollector.get_model_vars_dataframe()  # Time-series of global metrics
    agent_report = model.datacollector.get_agent_vars_dataframe()  # Time-series of agent-level data

    # -- Build a PyVis network from the final state
    net = build_pyvis_network(model)
    net_html = net.generate_html()

    # Store in session_state
    st.session_state["gd_model"] = model
    st.session_state["model_report_df"] = model_report
    st.session_state["agent_report_df"] = agent_report
    st.session_state["network_html"] = net_html


# If user clicks the Run button or if there's no model yet, run the model
if run_button or (st.session_state["gd_model"] is None):
    run_model()


# ------------------------------
# MAIN PAGE
# ------------------------------
st.title("Global Development Model with Network Visualization")

model_df = st.session_state["model_report_df"]
agent_df = st.session_state["agent_report_df"]

if model_df.empty or agent_df.empty:
    st.write("No model data. Click **Run** on the sidebar.")
else:
    # 1. Show the PyVis network
    st.subheader("Network Diagram")
    st.write("MacroAgents are gray squares; MicroAgents are circles (blue=employed, red=unemployed).")
    st.components.v1.html(st.session_state["network_html"], height=600)

    # 2. Charts of model-level time-series
    st.subheader("Global Metrics Over Time")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Plot each metric
    axes[0].plot(model_df.index, model_df["Global_Avg_Innovation"], label="Avg Innovation")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Innovation")
    axes[0].legend()

    axes[1].plot(model_df.index, model_df["Global_Avg_Resources"], label="Avg Resources")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Resources")
    axes[1].legend()

    axes[2].plot(model_df.index, model_df["Global_Employment_Rate"], label="Employment Rate")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Rate")
    axes[2].legend()

    axes[3].plot(model_df.index, model_df["Global_Gini_Resources"], label="Gini Coefficient")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Gini")
    axes[3].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 3. Final Snapshot Stats: Group by Country = "GDP", by Gender, by Race
    # Agent DataFrame is multi-indexed by (Step, AgentID). We'll extract the final step's data.
    final_step = agent_df.index.get_level_values("Step").max()
    final_agents = agent_df.xs(final_step, level="Step")

    st.subheader(f"Final Snapshot (Step {final_step}) Stats")

    # 3a. GDP by Country (sum of resources)
    st.write("**GDP by Country** (sum of Micro_Resources by country):")
    gdp_by_country = final_agents.groupby("Micro_CountryID")["Micro_Resources"].sum()
    st.write(gdp_by_country.to_frame("GDP (Resources Sum)"))

    # 3b. Average stats by Gender
    st.write("**Average Education, Resources, Innovation by Gender**")
    by_gender = final_agents.groupby("Micro_Gender")[["Micro_Education","Micro_Resources","Micro_Innovation"]].mean()
    st.write(by_gender)

    # 3c. Average stats by Race
    st.write("**Average Education, Resources, Innovation by Race**")
    by_race = final_agents.groupby("Micro_Race")[["Micro_Education","Micro_Resources","Micro_Innovation"]].mean()
    st.write(by_race)

    # You can add more tables or charts, e.g. by combining gender & race, or by country & gender, etc.
