"""
Run with:
  streamlit run server.py
in the terminal or command prompt.
Then open the local URL printed by Streamlit (e.g., http://localhost:8501).
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from model import GlobalDevelopmentModel


# ------------------------------
# STREAMLIT SIDEBAR CONTROLS
# ------------------------------
st.sidebar.title("Global Development Model Parameters")
num_countries = st.sidebar.slider(
    "Number of Countries",
    min_value=1, max_value=6,
    value=3, step=1
)
agents_per_country = st.sidebar.slider(
    "Agents per Country",
    min_value=5, max_value=100,
    value=30, step=5
)
innovation_factor = st.sidebar.slider(
    "Innovation Factor",
    min_value=0.0, max_value=0.1,
    value=0.03, step=0.01
)
steps_to_run = st.sidebar.number_input(
    "Steps to Run",
    min_value=1, max_value=200,
    value=50, step=1
)

run_button = st.sidebar.button("Run / Re-run Model")

# ------------------------------
# SESSION STATE FOR MODEL
# ------------------------------
# We store the model in Session State so it persists across UI interactions.
if "gd_model" not in st.session_state:
    st.session_state["gd_model"] = None
if "model_report_df" not in st.session_state:
    st.session_state["model_report_df"] = pd.DataFrame()


def run_model():
    """Creates and runs the GlobalDevelopmentModel, then stores the results."""
    model = GlobalDevelopmentModel(
        num_countries=num_countries,
        agents_per_country=agents_per_country,
        innovation_factor=innovation_factor
    )
    
    # Run the model for the specified number of steps
    for _ in range(int(steps_to_run)):
        model.step()

    # Extract the model-level DataCollector results
    model_report = model.datacollector.get_model_vars_dataframe()
    
    # Update session state
    st.session_state["gd_model"] = model
    st.session_state["model_report_df"] = model_report


# If user clicks the Run button or if there's no model in session, run it
if run_button or (st.session_state["gd_model"] is None):
    run_model()


# ------------------------------
# DISPLAY RESULTS
# ------------------------------
st.title("Global Development Model Results")

# Get the DataFrame of model reports
model_df = st.session_state["model_report_df"]
if model_df.empty:
    st.write("No model data yet. Click **Run** on the sidebar.")
else:
    st.write("Below is the final data from the model runs:")

    # Show the raw DataFrame
    st.dataframe(model_df)

    # Let's plot the timeseries for each metric
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes = axes.flatten()

    # 1) Global_Avg_Innovation
    axes[0].plot(model_df.index, model_df["Global_Avg_Innovation"], label="Avg Innovation")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Avg Innovation")
    axes[0].legend()

    # 2) Global_Avg_Resources
    axes[1].plot(model_df.index, model_df["Global_Avg_Resources"], label="Avg Resources")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Resources")
    axes[1].legend()

    # 3) Global_Employment_Rate
    axes[2].plot(model_df.index, model_df["Global_Employment_Rate"], label="Employment Rate")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Employment Rate")
    axes[2].legend()

    # 4) Global_Gini_Resources
    axes[3].plot(model_df.index, model_df["Global_Gini_Resources"], label="Gini Coefficient")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Gini")
    axes[3].legend()

    plt.tight_layout()
    st.pyplot(fig)
