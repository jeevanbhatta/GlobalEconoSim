import streamlit as st
# The set_page_config must be the first Streamlit command
st.set_page_config(page_title="Global Trade Simulation", layout="wide")

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import math
import random
import itertools

# Try to import statsmodels, but provide fallback if not available
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("""
        The statsmodels package is not installed. Some trend line features may not work.
        Please install it using: `pip install statsmodels`
    """)

from model import Country, TradeNetwork, simulate
from analytics import mfa_series, parameter_sensitivity_analysis
from visualization import plot_network, plot_histogram, plot_scatter, plot_line, plot_bar
# Import functions from trade_stats for country analysis
from trade_stats import (
    get_top_gdp_countries, get_bottom_gdp_countries,
    get_top_connected_countries, get_least_connected_countries,
    get_trade_efficiency_countries, get_growth_rate_countries,
    calculate_gdp_tariff_correlation, calculate_gdp_connections_correlation,
    get_shock_country_comparison
)

"""
Interactive Global Trade Simulation with MFA Contrast
----------------------------------------------------
This Streamlit application lets you explore how international trade dynamics
impact macro‑economic outcomes in a network of *n* countries. You can:

* Adjust the number of countries, simulation length, and initial policy levers.
* Introduce a targeted policy shock (e.g. tariff hike, export subsidy) to a
  single country and watch higher‑order spill‑overs propagate through the
  world economy.
* Compare full network simulation results to a **Mean‑Field Approximation**
  (MFA) that ignores network structure and treats every country as trading
  with the global average.
* Inspect headline indicators such as GDP growth, poverty headcount ratio,
  trade polarity (Herfindahl‑Hirschman Index of GDP share), and the Gini
  coefficient of GDP distribution.

The code is organised as follows:

1. 🏗 **Model primitives** (`Country`, `TradeNetwork`)
2. 🔄 **Simulation engine** (`simulate_trade`)
3. 📊 **Analytics** (world indicators, MFA functions)
4. 🎛 **Streamlit UI** (sidebar controls & visualisations)

Install requirements and run:
```bash
pip install streamlit networkx numpy pandas plotly~=5.19
streamlit run trade_app.py
```

**New in this version**
1. **MFA vs Simulation Time‑series** – headline indicators plotted side‑by‑side so you can watch divergence unfold.
2. **Interactive Network Graph** – live Plotly rendering of the trade web, node colours = political bloc, edge width = trade flow.
3. **Political/​Friendship Blocs** – pick `k` blocs; intra‑bloc tariffs & link probabilities differ from inter‑bloc values. A slider controls the intra‑vs‑inter tariff gap so you can see when the world fractures into components.
4. **Two‑Good World (A & B)** – each country draws random productivity for each good; gravity flows run per good and aggregate into GDP.
5. **Best & Worst Performers** - Compare top and bottom countries across different metrics like GDP, connectivity, trade efficiency and growth rate.
"""

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------

def continue_simulation(network, additional_steps, additional_shocks=None):
    """Continue an existing simulation for additional timesteps"""
    # Extract current parameters from network
    n = len(network.countries)
    blocs = max([c.bloc for c in network.countries.values()]) + 1
    
    # Estimate original parameters from the network
    # This is a simplified estimate as we don't store all original parameters
    conn_intra = sum(1 for u, v, d in network.G.edges(data=True) 
                     if network.countries[u].bloc == network.countries[v].bloc) / (n * (n-1)/2)
    conn_inter = sum(1 for u, v, d in network.G.edges(data=True)
                     if network.countries[u].bloc != network.countries[v].bloc) / (n * (n-1)/2)
    
    tariffs = [d['tariff'] for _, _, d in network.G.edges(data=True)]
    tariff_sd = np.std(tariffs) if tariffs else 0.05
    
    # Estimate intra-bloc and inter-bloc tariffs
    intra_tariffs = [d['tariff'] for u, v, d in network.G.edges(data=True) 
                    if network.countries[u].bloc == network.countries[v].bloc]
    inter_tariffs = [d['tariff'] for u, v, d in network.G.edges(data=True)
                    if network.countries[u].bloc != network.countries[v].bloc]
    
    tariff_intra_mu = np.mean(intra_tariffs) if intra_tariffs else 0.1
    tariff_inter_mu = np.mean(inter_tariffs) if inter_tariffs else 0.3
    tariff_gap = tariff_inter_mu - tariff_intra_mu
    
    # Check if it's a two-goods world
    two_goods = 'B' in next(iter(network.countries.values())).eff
    
    # Get the current history length
    current_steps = len(next(iter(network.countries.values())).history['gdp'])
    
    # Create a copy of the current network to continue from
    import copy
    net_copy = copy.deepcopy(network)
    
    # Run additional simulation steps
    for t in range(additional_steps):
        # Process any additional shocks
        if additional_shocks:
            for shock_id, shock_delta in additional_shocks:
                if 0 <= shock_id < n:
                    net_copy.apply_tariff_delta(shock_id, shock_delta)
        
        # Consider forming new blocs every 10 steps
        if (current_steps + t) % 10 == 0:
            net_copy.allow_bloc_formation(tariff_threshold=0.4)
        
        # Compute trade flows and update countries
        goods = ["A", "B"] if two_goods else ["A"]
        net_copy.compute_trade_flows(goods)
        
        for c in net_copy.countries.values():
            exports = min(c.exports, 1e12)
            imports = min(c.imports, 1e12)
            
            trade_balance = 0.3 * exports - 0.2 * imports
            raw_growth = trade_balance / max(c.gdp, 1e3)
            bounded_growth = 0.02 + 0.3 * np.tanh(raw_growth)
            
            c.gdp = min(c.gdp * (1 + bounded_growth), 1e15)
            
            try:
                growth_effect = np.clip(bounded_growth - 0.01, -0.5, 0.5)
                c.poverty_rate = max(c.poverty_rate * (1 - 0.3 * growth_effect), 0.01)
            except:
                c.poverty_rate = max(c.poverty_rate * 0.99, 0.01)
            
            c.log_step()
    
    return net_copy

def main():
    # Initialize session state at the beginning
    if 'network' not in st.session_state:
        st.session_state.network = None
    if 'params' not in st.session_state:
        st.session_state.params = {}
    if 'current_steps' not in st.session_state:
        st.session_state.current_steps = 0

    # Create tabs properly with individual references
    tab_names = ["Main Simulation", "Parameter Sensitivity Analysis", "MFA Comparison"]
    tab1, tab2, tab3 = st.tabs(tab_names)
    
    with tab1:
        st.title("🌐 Global Trade Simulation")
        st.markdown("""
        ## Tariff Basics
        - Definition: A tariff is a tax on imported goods, usually ad valorem (percentage of value).
        - Incidence: Tariffs raise consumer prices and generate government revenue, causing deadweight loss.
        - Retaliation: Countries may apply reciprocal tariffs, shown as purple dashed edges.
        - Network Effects: Tariffs on one link can ripple through trade partners.
        """)

        # Create a container for parameter explanations that can be expanded/collapsed
        with st.expander("📋 Parameter Explanations", expanded=False):
            st.markdown("""
            ### Simulation Parameters
            - **Countries**: Number of countries in the world economy.
            - **Political blocs**: Number of political or economic alliances (like EU, NAFTA).
            - **Time steps**: How many iterations to run the simulation.
            - **Intra‑bloc link prob**: Probability of trade links forming between countries in the same bloc.
            - **Inter‑bloc link prob**: Probability of trade links forming between countries in different blocs.
            - **Tariff gap**: How much higher tariffs are between countries in different blocs compared to same-bloc tariffs.
            - **Tariff SD**: Standard deviation in tariff rates (higher = more randomness in tariff policies).
            - **Two‑good world**: If checked, each country produces two goods with different productivities.
            - **Shock country id**: Which country will experience a policy shock.
            - **Δ tariff shock**: Size of tariff change to apply as a shock (positive = increase, negative = decrease).
            """)

        # sidebar with tooltips for parameter explanations
        n = st.sidebar.slider("Countries 🛈", 4, 80, 20, help="Number of countries in the simulation.")
        blocs = st.sidebar.slider("Political blocs 🛈", 0, min(15, n), 4, help="Number of political/economic alliances (like EU, NAFTA). Set to 0 for no blocs.")
        steps = st.sidebar.slider("Time steps 🛈", 20, 300, 60, 10, help="Number of simulation iterations. Higher values show longer-term effects but take more time.")
        conn_intra = st.sidebar.slider("Intra‑bloc link prob 🛈", 0.1, 1.0, 0.6, 0.05, help="Probability of trade links forming between countries in the same bloc. Higher values create denser within-bloc trade.")
        conn_inter = st.sidebar.slider("Inter‑bloc link prob 🛈", 0.0, 1.0, 0.2, 0.05, help="Probability of trade links forming between countries in different blocs. Lower values create more isolated blocs.")
        tariff_gap = st.sidebar.slider("Tariff gap (inter – intra) 🛈", 0.0, 0.9, 0.3, 0.05, help="How much higher tariffs are between countries in different blocs. Higher values represent stronger protectionism between blocs.")
        tariff_sd = st.sidebar.slider("Tariff SD 🛈", 0.0, 0.4, 0.05, 0.01, help="Standard deviation in tariff rates. Higher values create more variability in tariff policies.")
        two_goods = st.sidebar.checkbox("Two‑good world 🛈", value=True, help="If checked, each country produces two goods with different productivity levels, allowing for comparative advantage dynamics.")

        # Allow users to disable the shock
        apply_shock = st.sidebar.checkbox("Apply policy shock", value=True, help="If unchecked, no policy shock will be applied during the simulation.")
        
        if apply_shock:
            shock_id = st.sidebar.number_input("Shock country id 🛈", 0, n - 1, 0, help="Which country will experience a policy shock (tariff change).")
            shock_delta = st.sidebar.slider("Δ tariff shock 🛈", -0.5, 0.5, 0.1, 0.01, help="Size of tariff change (positive = increase, negative = decrease). Applied at 1/4 of the way through simulation.")
            policy_shock = (shock_id, shock_delta)
        else:
            policy_shock = None
            # Create hidden placeholder for shock parameters to maintain UI consistency
            st.sidebar.markdown("*Shock disabled*")

        # Add controls for continuing simulation
        if st.session_state.network is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Continue Simulation")
            additional_steps = st.sidebar.slider("Additional steps", 10, 100, 30, 10, 
                                               help="How many more time steps to run from current state")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                run = st.button("New Run 🚀")
            with col2:
                continue_run = st.button("Continue 🔄")
        else:
            run = st.sidebar.button("Run 🚀")
            continue_run = False
            additional_steps = 30  # Default value

        # --- detect if any inputs have changed since last run ---
        current_params = {
            'n': n, 'blocs': blocs, 'conn_intra': conn_intra, 'conn_inter': conn_inter,
            'tariff_gap': tariff_gap, 'tariff_sd': tariff_sd,
            'two_goods': two_goods, 'policy_shock': policy_shock
        }
        params_changed = st.session_state.params != current_params

        if not run and not continue_run and st.session_state.network is None:
            st.info("Set parameters and press *Run*.")
        else:
            # If this is a new run or params changed, create a new simulation
            if run or st.session_state.network is None or params_changed:
                # Store current parameters
                st.session_state.params = current_params
                
                # Run new simulation
                net = simulate(
                    n, blocs, steps, conn_intra, conn_inter, tariff_gap, tariff_sd, two_goods,
                    policy_shock=policy_shock,
                )
                st.session_state.network = net
                st.session_state.current_steps = steps
            
            # If this is a continue run, extend the simulation
            elif continue_run:
                # Continue simulation with additional steps
                current_net = st.session_state.network
                
                # Run additional steps with the continue_simulation method
                with st.spinner(f"Continuing simulation for {additional_steps} steps..."):
                    net = continue_simulation(
                        current_net, additional_steps, 
                        additional_shocks=None  # Add option for additional shocks if needed
                    )
                    
                    st.session_state.network = net
                    st.session_state.current_steps += additional_steps
                    
                    # Show a success message
                    st.success(f"Simulation continued for {additional_steps} more steps. Total steps: {st.session_state.current_steps}")
            else:
                # Just use the existing network from session state
                net = st.session_state.network

            # ---------------- Indicators ------------------------------------
            total_steps = st.session_state.current_steps
            world_gdp_series = [sum(c.history["gdp"][t] for c in net.countries.values()) for t in range(total_steps)]
            mfa_series_vals = mfa_series(net, total_steps)

            ind_df = pd.DataFrame({"Simulation": world_gdp_series, "MFA": mfa_series_vals})
            st.subheader("World GDP – Simulation vs MFA")
            st.line_chart(ind_df)

            # Tariff Distribution and Correlation
            tariffs = [d['tariff'] for _, _, d in net.G.edges(data=True)]
            flows = [d['flow'] for _, _, d in net.G.edges(data=True)]
            if tariffs:
                st.subheader("Tariff Distribution")
                hist_fig = plot_histogram(tariffs, nbins=20, title="Tariffs Across All Trade Links")
                st.plotly_chart(hist_fig, use_container_width=True)

                st.subheader("Tariff vs Trade Flow")
                df_tf = pd.DataFrame({'Tariff': tariffs, 'Flow': flows})
                scatter_fig = plot_scatter(df_tf, x='Tariff', y='Flow', trendline='ols', title="Tariff vs Trade Flow")
                st.plotly_chart(scatter_fig, use_container_width=True)

            # fragmentation: components after pruning weak edges
            G_copy = net.G.copy()
            flows = []
            for *_, d in G_copy.edges(data=True):
                if "flow" in d:
                    flows.append(d["flow"])

            if flows:  # Only proceed if there are flows to analyze
                flows_array = np.array(flows)
                threshold = np.percentile(flows_array, 40)  # weakest 40% removed
                weak_edges = [(u, v) for u, v, d in G_copy.edges(data=True) if d.get("flow", 0) < threshold]
                G_copy.remove_edges_from(weak_edges)
                comps = nx.number_weakly_connected_components(G_copy)
                st.write(f"**Fragments after weak‑edge removal:** {comps}")
            else:
                st.write("**No trade flows to analyze for fragmentation**")

            # network viz
            st.subheader("Trade Network (edge width = flow, node colour = bloc)")
            st.plotly_chart(plot_network(net), use_container_width=True)
            
            # Display top and bottom countries by GDP and trade connections
            st.subheader("Country Analysis")
            
            # Community detection visualization
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(net.G.to_undirected()))
                sizes = [len(c) for c in communities]
                df_comm = pd.DataFrame({'Community': range(1, len(sizes)+1), 'Size': sizes})
                df_comm = df_comm.sort_values('Size', ascending=False).reset_index(drop=True)
                st.bar_chart(df_comm.set_index('Community')['Size'])
                st.write(f"Detected {len(communities)} communities. Largest size: {max(sizes)}")
            except Exception:
                st.write("Community detection failed. Ensure networkx >=2.5 is installed.")
                
            # Best and worst performers analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 5 Countries by GDP")
                top_gdp_df = get_top_gdp_countries(net, 5)
                st.dataframe(top_gdp_df)
                
                st.subheader("Top 5 Most Connected Countries")
                top_conn_df = get_top_connected_countries(net, 5)
                st.dataframe(top_conn_df)
                
                st.subheader("Top 5 Countries by Trade Efficiency")
                top_eff_df = get_trade_efficiency_countries(net, 5, reverse=False)
                st.dataframe(top_eff_df)
            
            with col2:
                st.subheader("Bottom 5 Countries by GDP")
                bottom_gdp_df = get_bottom_gdp_countries(net, 5)
                st.dataframe(bottom_gdp_df)
                
                st.subheader("5 Least Connected Countries")
                bottom_conn_df = get_least_connected_countries(net, 5)
                st.dataframe(bottom_conn_df)
                
                st.subheader("Top 5 Countries by Growth Rate")
                growth_df = get_growth_rate_countries(net, 5, highest=True)
                st.dataframe(growth_df)
            
            # Correlation analysis
            st.subheader("Economic Correlations")
            tariff_corr, tariff_gdps, tariff_data = calculate_gdp_tariff_correlation(net)
            conn_corr, conn_gdps, conn_data = calculate_gdp_connections_correlation(net)
            
            corr_data = {
                'Metric': ['GDP vs Tariff Level', 'GDP vs Connection Count'],
                'Correlation': [tariff_corr, conn_corr]
            }
            corr_df = pd.DataFrame(corr_data)
            
            # Plot correlation bar chart
            fig_corr = plot_bar(
                corr_df, 
                x='Metric', 
                y='Correlation', 
                title="Economic Correlations",
                barmode='group'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Shock impact analysis if a shock was applied
            if policy_shock is not None and apply_shock:
                st.subheader(f"Impact of Tariff Shock on Country {shock_id}")
                shock_comp_df = get_shock_country_comparison(net, shock_id)
                if shock_comp_df is not None:
                    st.dataframe(shock_comp_df)
                    
                    # Visualize growth comparison
                    growth_comp = pd.DataFrame({
                        'Entity': ['Shocked Country', 'Neighbors', 'World Average'],
                        'Growth Rate': shock_comp_df['Growth Rate'].values
                    })
                    fig_shock = plot_bar(
                        growth_comp, 
                        x='Entity', 
                        y='Growth Rate',
                        title="Growth Rate Comparison",
                        color='Entity'
                    )
                    st.plotly_chart(fig_shock, use_container_width=True)
                else:
                    st.write("No data available for shock comparison.")

            # GDP histogram
            gdp_vals = [c.gdp for c in net.countries.values()]
            st.subheader("GDP Distribution")
            # Render GDP histogram using Plotly Express
            fig_hist = plot_histogram(gdp_vals, nbins=20, title="GDP Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

            # poverty trend
            avg_pov = np.mean([c.history["poverty_rate"] for c in net.countries.values()], axis=0)
            st.subheader("Average Poverty Rate")
            
            # Convert to DataFrame for better formatting
            poverty_df = pd.DataFrame({
                'Time Step': range(len(avg_pov)),
                'Poverty Rate': avg_pov
            })
            fig_poverty = px.line(poverty_df, x='Time Step', y='Poverty Rate', 
                                title="Average Poverty Rate Over Time")
            st.plotly_chart(fig_poverty, use_container_width=True)
            
            # Gini coefficient calculation and visualization
            st.subheader("Inequality (Gini Coefficient)")
            gini_series = []
            for t in range(total_steps):
                gdp_values = sorted([c.history["gdp"][t] for c in net.countries.values() if t < len(c.history["gdp"])])
                n_countries = len(gdp_values)
                if n_countries > 1:
                    cum_wealth = np.cumsum(gdp_values)
                    gini = (n_countries + 1 - 2 * np.sum(cum_wealth) / cum_wealth[-1]) / n_countries
                else:
                    gini = 0
                gini_series.append(gini)
            
            gini_df = pd.DataFrame({
                'Time Step': range(len(gini_series)),
                'Gini Coefficient': gini_series
            })
            
            fig_gini = px.line(gini_df, x='Time Step', y='Gini Coefficient',
                             title="Inequality Over Time (Gini Coefficient)")
            st.plotly_chart(fig_gini, use_container_width=True)

            st.markdown("""
            ### Insights
            * **Tariff gap:** When the inter‑bloc tariffs exceed intra by ~0.4+, the network often splits into isolated blocs (see *Fragments* count).
            * **MFA Divergence:** Mean‑field under‑reports both volatility and concentration; network effects amplify inequality in many runs.
            * **Two‑good heterogeneity** creates comparative‑advantage cycles – some blocs specialise in A, others in B – further distancing results from MFA's uniform world.
            * **Network effects:** Countries with higher connectivity tend to experience faster growth and lower poverty rates.
            * **Shock propagation:** Tariff shocks create ripple effects that can be observed in neighboring countries' growth trajectories.
            """)

    with tab2:
        st.title("📈 Parameter Sensitivity Analysis")
        st.markdown("""
        Explore how changing one parameter affects the simulation outcomes while keeping all other parameters fixed.
        This helps identify critical thresholds and non-linear effects in the trade network.
        """)
        
        # Add explanation for error bars and replicates
        with st.expander("ℹ️ About Sensitivity Analysis & Error Bars", expanded=False):
            st.markdown("""
            ### Understanding the Analysis
            
            This tab runs multiple simulations while varying just one parameter value at a time,
            keeping all other parameters fixed. This approach reveals how sensitive outcomes are
            to changes in each parameter.
            
            ### About Error Bars & Replicates
            
            **Error bars** show the standard deviation across multiple simulation runs with identical 
            parameters but different random seeds. Larger error bars indicate:
            
            - Higher variability in outcomes
            - Less predictable system behavior
            - Potential chaotic dynamics at certain parameter values
            
            **Replicates** are multiple identical simulations that differ only in their random initialization.
            More replicates provide:
            
            - Greater confidence in results
            - Better understanding of inherent system variability
            - Ability to identify truly stable parameter regions
            
            The **coefficient of variation** (CV) shown in the Equilibrium Assessment measures the relative 
            variability between replicates. A low CV suggests the system reached equilibrium.
            """)
        
        # Parameter selection
        param_name = st.selectbox(
            "Parameter to vary", 
            ["tariff_gap", "n", "blocs", "conn_intra", "conn_inter", "tariff_sd"],
            format_func=lambda x: {
                "tariff_gap": "Tariff Gap (inter-bloc vs intra-bloc)",
                "n": "Number of Countries",
                "blocs": "Number of Political Blocs",
                "conn_intra": "Intra-bloc Connectivity",
                "conn_inter": "Inter-bloc Connectivity",
                "tariff_sd": "Tariff Standard Deviation"
            }.get(x, x)
        )
        
        # Target metric selection
        target_metric = st.selectbox(
            "Target metric", 
            ["world_gdp", "poverty", "trade_volume", "fragmentation", "gini"],
            format_func=lambda x: {
                "world_gdp": "World GDP",
                "poverty": "Poverty Rate",
                "trade_volume": "Trade Volume",
                "fragmentation": "Network Fragmentation",
                "gini": "Gini Coefficient"
            }.get(x, x)
        )
        
        # Parameter range
        if param_name == "tariff_gap":
            param_range = np.linspace(0.0, 0.8, 9)
        elif param_name == "n":
            param_range = [10, 15, 20, 25, 30, 40, 50]
        elif param_name == "blocs":
            param_range = list(range(2, 11))
        elif param_name in ["conn_intra", "conn_inter"]:
            param_range = np.linspace(0.1, 0.9, 9)
        elif param_name == "tariff_sd":
            param_range = np.linspace(0.01, 0.3, 8)
        else:
            param_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Number of steps
        steps = st.slider("Simulation steps", 20, 200, 60, 10)
        
        # Fixed parameters (default values)
        fixed_params = {
            'n': 20,
            'blocs': 4,
            'conn_intra': 0.6,
            'conn_inter': 0.2,
            'tariff_gap': 0.3,
            'tariff_sd': 0.05,
            'two_goods': True
        }
        
        # Add replicates option for robustness
        num_replicates = st.slider("Number of simulation replicates", 1, 10, 3, 
                                  help="More replicates give more reliable results but take longer to compute")
        
        # Run analysis button
        if st.button("Run Sensitivity Analysis", key="run_sens_analysis"):
            with st.spinner(f"Running sensitivity analysis for {param_name}..."):
                results_df = parameter_sensitivity_analysis(
                    param_name, param_range, fixed_params, target_metric, steps, num_replicates
                )
                
                # Get display-friendly parameter name
                param_display_name = {
                    "tariff_gap": "Tariff Gap",
                    "n": "Number of Countries",
                    "blocs": "Number of Political Blocs",
                    "conn_intra": "Intra-bloc Connectivity",
                    "conn_inter": "Inter-bloc Connectivity",
                    "tariff_sd": "Tariff Standard Deviation"
                }.get(param_name, param_name.replace('_', ' ').title())
                
                # Get display-friendly metric name
                metric_display_name = {
                    "world_gdp": "World GDP",
                    "poverty": "Poverty Rate",
                    "trade_volume": "Trade Volume", 
                    "fragmentation": "Network Fragmentation",
                    "gini": "Gini Coefficient"
                }.get(target_metric, target_metric.replace('_', ' ').title())
                
                # Line chart with error bars for final values
                st.subheader(f"Effect of {param_display_name} on {metric_display_name}")
                
                # Plot with error bars using Plotly
                fig = plot_line(
                    results_df, 
                    x="param_value", 
                    y="final_value",
                    error_y="final_value_std",
                    markers=True,
                    title=f"{metric_display_name} vs {param_display_name}"
                )
                
                # Update axis labels explicitly
                fig.update_layout(
                    xaxis_title=param_display_name,
                    yaxis_title=metric_display_name
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find critical values and turning points
                try:
                    turning_points = []
                    values = results_df["final_value"].values
                    for i in range(1, len(values)-1):
                        if (values[i] > values[i-1] and values[i] > values[i+1]) or \
                           (values[i] < values[i-1] and values[i] < values[i+1]):
                            turning_points.append((results_df["param_value"].iloc[i], values[i]))
                    
                    if turning_points:
                        st.write("**Critical values detected:**")
                        for point in turning_points:
                            if target_metric in ["poverty", "gini", "fragmentation"]:
                                direction = "minimum" if point[1] < values[0] and point[1] < values[-1] else "maximum"
                            else:
                                direction = "maximum" if point[1] > values[0] and point[1] > values[-1] else "minimum"
                            st.write(f"• {param_display_name} = {point[0]:.2f} ({direction} {metric_display_name}: {point[1]:.2f})")
                    
                    # Add explanation of what the error bars represent
                    if num_replicates > 1:
                        st.info(f"""
                        **About the error bars:** Each point shows the mean value across {num_replicates} simulation runs.
                        Error bars show ±1 standard deviation, representing the inherent variability in outcomes 
                        due to randomness in the simulation. Larger error bars indicate parameter values where outcomes 
                        are less predictable.
                        """)
                        
                    # Identify optimal parameter value
                    optimal_idx = results_df["final_value"].idxmax() if target_metric in ["world_gdp", "trade_volume"] else results_df["final_value"].idxmin()
                    optimal_value = results_df["param_value"].iloc[optimal_idx]
                    st.success(f"**Optimal {param_display_name}:** {optimal_value:.2f} for {'maximizing' if target_metric in ['world_gdp', 'trade_volume'] else 'minimizing'} {metric_display_name}")
                except Exception as e:
                    st.warning(f"Could not identify critical points due to: {str(e)}")
                
                # Bar chart of growth rates (if applicable)
                if target_metric in ["world_gdp", "poverty", "trade_volume"]:
                    fig2 = px.bar(
                        results_df, 
                        x="param_value", 
                        y="growth_rate",
                        error_y="growth_rate_std",
                        title=f"Growth Rate in {metric_display_name} vs {param_display_name}"
                    )
                    
                    # Update axis labels explicitly
                    fig2.update_layout(
                        xaxis_title=param_display_name,
                        yaxis_title=f"{metric_display_name} Growth Rate"
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    if num_replicates > 1:
                        st.info("""
                        **Interpreting growth rates:** These bars show the percentage change from start to end of simulation.
                        Error bars indicate variability across replicates. Consistent growth/decline across parameter values
                        suggests a robust relationship, while high variability suggests parameter-dependent dynamics.
                        """)
                
                # Equilibrium assessment
                if num_replicates > 1:
                    st.subheader("Equilibrium Assessment")
                    cv = results_df["final_value_std"] / results_df["final_value"] * 100  # coefficient of variation
                    mean_cv = np.mean(cv)
                    st.write(f"**Average coefficient of variation:** {mean_cv:.2f}%")
                    
                    if mean_cv < 5:
                        st.success("✅ Low variability between replicates suggests the system reached equilibrium.")
                    elif mean_cv < 15:
                        st.info("⚠️ Moderate variability between replicates. Consider increasing simulation steps.")
                    else:
                        st.warning("❌ High variability between replicates. The system likely does not reach equilibrium with current settings.")
                    
                    # Show parameter regions with high variability
                    high_cv_idx = cv > 15
                    if any(high_cv_idx):
                        high_var_params = results_df.loc[high_cv_idx, "param_value"].values
                        st.write("**Parameter values with high outcome variability:**")
                        for p in high_var_params:
                            st.write(f"• {param_display_name} = {p:.2f}")
                
                # Data table
                st.subheader("Detailed Results")
                
                # Format DataFrame for display with better column names
                display_df = results_df[["param_value", "final_value", "final_value_std", "growth_rate", "growth_rate_std"]].copy()
                display_df.columns = [
                    param_display_name, 
                    f"Final {metric_display_name}", 
                    f"{metric_display_name} Std Dev", 
                    "Growth Rate (%)", 
                    "Growth Rate Std Dev"
                ]
                
                # Format values for better readability
                if target_metric == "world_gdp":
                    display_df[f"Final {metric_display_name}"] = display_df[f"Final {metric_display_name}"].map(lambda x: f"{x:,.0f}")
                    display_df[f"{metric_display_name} Std Dev"] = display_df[f"{metric_display_name} Std Dev"].map(lambda x: f"{x:,.0f}")
                else:
                    display_df[f"Final {metric_display_name}"] = display_df[f"Final {metric_display_name}"].map(lambda x: f"{x:.3f}")
                    display_df[f"{metric_display_name} Std Dev"] = display_df[f"{metric_display_name} Std Dev"].map(lambda x: f"{x:.3f}")
                
                display_df["Growth Rate (%)"] = display_df["Growth Rate (%)"].map(lambda x: f"{x*100:.1f}%")
                display_df["Growth Rate Std Dev"] = display_df["Growth Rate Std Dev"].map(lambda x: f"{x*100:.1f}%")
                
                st.dataframe(display_df)
    
    with tab3:
        st.title("📊 Mean-Field Approximation vs Network Simulation")
        st.markdown("""
        ### What is Mean-Field Approximation (MFA)?
        
        In complex systems modeling, a mean-field approximation replaces all interactions between individual elements 
        with an average or "mean" interaction field. For our trade network:
        
        - **Full Simulation**: Each country has specific bilateral trade relationships
        - **MFA**: Each country interacts with an "average world economy"
        
        The comparison shows how network structure and heterogeneity create outcomes that differ from mean-field predictions.
        """)
        
        # Parameters for the MFA comparison
        st.subheader("Run Comparison Simulation")
        col1, col2 = st.columns(2)
        
        with col1:
            mfa_countries = st.slider("Number of countries", 5, 50, 15, step=5)
            mfa_blocs = st.slider("Number of blocs", 2, 10, 3)
            mfa_steps = st.slider("Simulation steps", 20, 200, 80)
        
        with col2:
            mfa_tariff_gap = st.slider("Tariff gap", 0.0, 0.8, 0.3, 0.1)
            mfa_conn_intra = st.slider("Intra-bloc connectivity", 0.1, 1.0, 0.7, 0.1) 
            include_shock = st.checkbox("Include policy shock", value=True)
        
        shock_params = (0, 0.2) if include_shock else None
        
        if st.button("Run MFA Comparison", key="run_mfa_comparison"):
            with st.spinner("Running simulation..."):
                # Run full network simulation
                net = simulate(
                    n=mfa_countries, 
                    blocs=mfa_blocs, 
                    steps=mfa_steps,
                    conn_intra=mfa_conn_intra,
                    conn_inter=0.2,
                    tariff_gap=mfa_tariff_gap,
                    tariff_sd=0.05,
                    two_goods=True,
                    policy_shock=shock_params
                )
                
                # Extract time series data
                world_gdp = [sum(c.history["gdp"][t] for c in net.countries.values()) for t in range(mfa_steps)]
                avg_poverty = [np.mean([c.history["poverty_rate"][t] for c in net.countries.values()]) for t in range(mfa_steps)]
                total_exports = [sum(c.history["exports"][t] for c in net.countries.values() if t < len(c.history["exports"])) for t in range(mfa_steps)]
                
                # Calculate Gini coefficient for each time step
                gini_series = []
                for t in range(mfa_steps):
                    gdp_values = sorted([c.history["gdp"][t] for c in net.countries.values() if t < len(c.history["gdp"])])
                    n = len(gdp_values)
                    if n > 1:
                        cum_wealth = np.cumsum(gdp_values)
                        gini = (n + 1 - 2 * np.sum(cum_wealth) / cum_wealth[-1]) / n
                    else:
                        gini = 0
                    gini_series.append(gini)
                
                # Generate MFA predictions
                mfa_gdp = mfa_series(net, mfa_steps)
                
                # Create MFA poverty series (simplified model)
                initial_poverty = avg_poverty[0]
                mfa_poverty = []
                for t in range(mfa_steps):
                    mfa_growth_factor = mfa_gdp[t] / mfa_gdp[0] if t > 0 else 1
                    poverty_reduction = 0.3 * (mfa_growth_factor - 1)  # simplified model
                    mfa_poverty.append(max(initial_poverty * (1 - poverty_reduction), 0.01))
                
                # Create MFA gini series (assume constant in MFA)
                mfa_gini = [gini_series[0]] * mfa_steps
                
                # Bundle data for plotting
                df_gdp = pd.DataFrame({
                    'Step': range(mfa_steps),
                    'Network Simulation': world_gdp,
                    'Mean-Field Approximation': mfa_gdp
                })
                
                df_poverty = pd.DataFrame({
                    'Step': range(mfa_steps),
                    'Network Simulation': avg_poverty,
                    'Mean-Field Approximation': mfa_poverty
                })
                
                df_gini = pd.DataFrame({
                    'Step': range(mfa_steps),
                    'Network Simulation': gini_series,
                    'Mean-Field Approximation': mfa_gini
                })
                
                # Plot the comparisons
                st.subheader("World GDP: Network vs MFA")
                fig_gdp = plot_line(df_gdp, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
                                title="World GDP Over Time")
                st.plotly_chart(fig_gdp, use_container_width=True)
                
                # Plot divegence metric - difference between simulation and MFA
                df_gdp['Divergence'] = df_gdp['Network Simulation'] - df_gdp['Mean-Field Approximation']
                fig_div = px.area(df_gdp, x='Step', y='Divergence', 
                                title="Network-MFA Divergence (GDP)")
                st.plotly_chart(fig_div, use_container_width=True)
                
                # Poverty rate comparison
                st.subheader("Poverty Rate: Network vs MFA")
                fig_pov = plot_line(df_poverty, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
                                title="Average Poverty Rate Over Time")
                st.plotly_chart(fig_pov, use_container_width=True)
                
                # Gini coefficient comparison
                st.subheader("Inequality (Gini): Network vs MFA")
                fig_gini = plot_line(df_gini, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
                                title="Gini Coefficient Over Time")
                st.plotly_chart(fig_gini, use_container_width=True)
                
                # Analysis of results
                divergence_pct = abs(df_gdp['Divergence'].iloc[-1] / df_gdp['Mean-Field Approximation'].iloc[-1] * 100)
                
                st.subheader("Key Insights")
                st.write(f"Final divergence between simulation and MFA: {divergence_pct:.1f}%")
                
                # Show summary statistics
                if divergence_pct > 15:
                    st.success("Strong network effects detected! The full simulation diverges significantly from mean-field predictions.")
                    st.markdown("""
                    **Implications for Tariff Policy:**
                    - Network structure amplifies the effects of tariff changes
                    - Targeted tariffs on central nodes have outsized impact
                    - Political blocs create more resilience against external tariffs
                    """)
                else:
                    st.info("Weak network effects observed. In this configuration, mean-field approximation captures most dynamics.")
                    st.markdown("""
                    **Implications for Tariff Policy:**
                    - Uniform tariff changes may be well-predicted by simpler models
                    - Network centrality matters less in this configuration
                    - Political bloc structure has limited impact on outcomes
                    """)
                
                # Visualization of final state network
                st.subheader("Final Trade Network State")
                st.plotly_chart(plot_network(net), use_container_width=True)


if __name__ == "__main__":
    main()
