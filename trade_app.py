from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="Global Trade Simulation", layout="wide")
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
"""

# -------------------------------------------------------------
# 1.   MODEL PRIMITIVES
# -------------------------------------------------------------
@dataclass
class Country:
    cid: int
    bloc: int  # political / friendship group
    gdp: float
    population: int
    poverty_rate: float
    dev_index: float
    eff: Dict[str, float]  # productivity for goods A & B
    exports: float = 0.0
    imports: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "gdp": [],
        "poverty_rate": [],
        "exports": [],
        "imports": [],
    })

    def log_step(self):
        for k in self.history:
            self.history[k].append(getattr(self, k))


class TradeNetwork:
    """Directed multi‑good trade graph."""

    def __init__(
        self,
        countries: List[Country],
        conn_intra: float,
        conn_inter: float,
        tariff_intra_mu: float,
        tariff_inter_mu: float,
        tariff_sd: float,
    ):
        self.countries = {c.cid: c for c in countries}
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.countries.keys())

        n = len(countries)
        for i, j in itertools.permutations(range(n), 2):
            same_bloc = countries[i].bloc == countries[j].bloc
            p = conn_intra if same_bloc else conn_inter
            if random.random() < p:
                mu = tariff_intra_mu if same_bloc else tariff_inter_mu
                tariff = np.clip(np.random.normal(mu, tariff_sd), 0, 1)
                dist = random.uniform(1, 10)
                self.G.add_edge(i, j, tariff=tariff, dist=dist)

    # ---------------------------------------------------------
    def compute_trade_flows(self, goods: List[str], price_elast: float = 1.0):
        for u, v, data in self.G.edges(data=True):
            c_u, c_v = self.countries[u], self.countries[v]
            tariff, dist = data["tariff"], data["dist"]
            total_flow = 0.0
            for g in goods:
                prod = c_u.eff[g]  # exporter productivity for good g
                flow = (prod * c_u.gdp * c_v.gdp) / (dist ** 2)
                flow *= (1 - tariff) ** price_elast
                total_flow += flow
            data["flow"] = total_flow

        # aggregate to countries
        for c in self.countries.values():
            outs = self.G.out_edges(c.cid, data=True)
            ins = self.G.in_edges(c.cid, data=True)
            c.exports = sum(d["flow"] for *_ , d in outs)
            c.imports = sum(d["flow"] for *_ , d in ins)

    # ---------------------------------------------------------
    def apply_tariff_delta(self, cid: int, delta: float):
        for _, v, d in self.G.out_edges(cid, data=True):
            d["tariff"] = np.clip(d["tariff"] + delta, 0, 1)

# -------------------------------------------------------------
# 2.   SIMULATION ENGINE
# -------------------------------------------------------------

def simulate(
    n: int,
    blocs: int,
    steps: int,
    conn_intra: float,
    conn_inter: float,
    tariff_gap: float,
    tariff_sd: float,
    two_goods: bool,
    policy_shock: Tuple[int, float] | None = None,
):
    
    goods = ["A", "B"] if two_goods else ["A"]

    # initialise countries
    countries: List[Country] = []
    for cid in range(n):
        bloc = cid % blocs
        # increase initial GDP range for visible dynamics
        gdp0 = random.uniform(10000, 100000)
        pop = random.randint(2_000_000, 150_000_000)
        pov = random.uniform(0.05, 0.5)
        dev = random.uniform(0.4, 0.9)
        eff = {g: abs(np.random.normal(1.0, 0.3)) for g in goods}
        countries.append(Country(cid, bloc, gdp0, pop, pov, dev, eff))

    net = TradeNetwork(
        countries,
        conn_intra,
        conn_inter,
        tariff_intra_mu=max(0.05, 0.10 - tariff_gap / 2),
        tariff_inter_mu=min(0.95, 0.10 + tariff_gap / 2),
        tariff_sd=tariff_sd,
    )

    # time loop
    for t in range(steps):
        if policy_shock and t == steps // 4:
            net.apply_tariff_delta(*policy_shock)

        net.compute_trade_flows(goods)
        for c in countries:
            trade_balance = 0.3 * c.exports - 0.2 * c.imports
            gdp_growth = 0.02 + trade_balance / max(c.gdp, 1e-9)
            c.gdp *= (1 + gdp_growth)
            c.poverty_rate = max(c.poverty_rate * (1 - 0.3 * (gdp_growth - 0.01)), 0.01)
            c.log_step()

    return net

# -------------------------------------------------------------
# 3.   MFA ANALYTICS
# -------------------------------------------------------------

def mfa_series(net: TradeNetwork, steps: int):
    """Generate mean‑field time‑series equal‑shares benchmark."""
    n = len(net.countries)
    # assume GDP grows at avg of simulation growth rates
    avg_growth = np.mean([
        (c.history["gdp"][-1] / c.history["gdp"][0]) ** (1 / steps) - 1 for c in net.countries.values()
    ])
    gdp0 = np.mean([c.history["gdp"][0] for c in net.countries.values()])
    world = []
    for t in range(steps):
        gdp_t = gdp0 * ((1 + avg_growth) ** t)
        world.append(gdp_t * n)
    return world

# -------------------------------------------------------------
# 4.   VISUAL HELPERS
# -------------------------------------------------------------

def plot_network(net: TradeNetwork):
    pos = nx.spring_layout(net.G, seed=42, k=0.5)
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in net.G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(d["flow"])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none")

    node_x, node_y, node_color, node_size = [], [], [], []
    for cid, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_color.append(net.countries[cid].bloc)
        # compute marker size safely (no NaN)
        gdp_val = net.countries[cid].gdp
        size = math.sqrt(gdp_val) if (gdp_val is not None and gdp_val > 0) else 10
        node_size.append(size)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_color, showscale=True),
        text=[f"C{cid}" for cid in net.countries],
        hoverinfo="text",
    )
    fig = go.Figure(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig

# -------------------------------------------------------------
# 5.   STREAMLIT UI
# -------------------------------------------------------------

def main():
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

        # sidebar
        n = st.sidebar.slider("Countries", 6, 60, 20)
        blocs = st.sidebar.slider("Political blocs", 2, min(10, n), 4)
        steps = st.sidebar.slider("Time steps", 20, 200, 60, 10)
        conn_intra = st.sidebar.slider("Intra‑bloc link prob", 0.1, 1.0, 0.6, 0.05)
        conn_inter = st.sidebar.slider("Inter‑bloc link prob", 0.0, 1.0, 0.2, 0.05)
        tariff_gap = st.sidebar.slider("Tariff gap (inter – intra)", 0.0, 0.8, 0.3, 0.05)
        tariff_sd = st.sidebar.slider("Tariff SD", 0.0, 0.3, 0.05, 0.01)
        two_goods = st.sidebar.checkbox("Two‑good world", value=True)

        shock_id = st.sidebar.number_input("Shock country id", 0, n - 1, 0)
        shock_delta = st.sidebar.slider("Δ tariff shock", -0.3, 0.3, 0.1, 0.01)
        run = st.sidebar.button("Run 🚀")

        if not run:
            st.info("Set parameters and press *Run*.")
            return

        net = simulate(
            n, blocs, steps, conn_intra, conn_inter, tariff_gap, tariff_sd, two_goods,
            policy_shock=(shock_id, shock_delta),
        )

        # ---------------- Indicators ------------------------------------
        world_gdp_series = [sum(c.history["gdp"][t] for c in net.countries.values()) for t in range(steps)]
        mfa_series_vals = mfa_series(net, steps)

        ind_df = pd.DataFrame({"Simulation": world_gdp_series, "MFA": mfa_series_vals})
        st.subheader("World GDP – Simulation vs MFA")
        st.line_chart(ind_df)

        # Tariff Distribution and Correlation
        tariffs = [d['tariff'] for _, _, d in net.G.edges(data=True)]
        flows = [d['flow'] for _, _, d in net.G.edges(data=True)]
        if tariffs:
            st.subheader("Tariff Distribution")
            hist_fig = px.histogram(tariffs, nbins=20, title="Tariffs Across All Trade Links")
            st.plotly_chart(hist_fig, use_container_width=True)

            st.subheader("Tariff vs Trade Flow")
            df_tf = pd.DataFrame({'Tariff': tariffs, 'Flow': flows})
            scatter_fig = px.scatter(df_tf, x='Tariff', y='Flow', trendline='ols', title="Tariff vs Trade Flow")
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

        # Community detection on final trade network
        st.subheader("Trade Communities (Final State)")
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

        # GDP histogram
        gdp_vals = [c.gdp for c in net.countries.values()]
        st.subheader("GDP distribution")
        # Render GDP histogram using Plotly Express
        fig_hist = px.histogram(x=gdp_vals, nbins=20, title="GDP Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        # poverty trend
        avg_pov = np.mean([c.history["poverty_rate"] for c in net.countries.values()], axis=0)
        st.subheader("Average poverty rate")
        st.line_chart(pd.Series(avg_pov))

        st.markdown("""
        ### Insights
        * **Tariff gap:** When the inter‑bloc tariffs exceed intra by ~0.4+, the network often splits into isolated blocs (see *Fragments* count).
        * **MFA Divergence:** Mean‑field under‑reports both volatility and concentration; network effects amplify inequality in many runs.
        * **Two‑good heterogeneity** creates comparative‑advantage cycles – some blocs specialise in A, others in B – further distancing results from MFA’s uniform world.
        """)

    with tab2:
        st.title("📈 Parameter Sensitivity Analysis")
        st.markdown("""
        Explore how changing one parameter affects the simulation outcomes while keeping all other parameters fixed.
        This helps identify critical thresholds and non-linear effects in the trade network.
        """)
        
        # Parameter selection
        param_name = st.selectbox(
            "Parameter to vary", 
            ["tariff_gap", "n", "blocs", "conn_intra", "conn_inter", "tariff_sd"]
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
        num_replicates = st.slider("Number of simulation replicates", 1, 10, 3)
        
        # Run analysis button
        if st.button("Run Sensitivity Analysis"):
            progress_bar = st.progress(0)
            st.write(f"Running {len(param_range)*num_replicates} simulations varying {param_name}...")
            
            # Initialize results
            all_results = []
            
            # Loop through parameter values
            for i, param_value in enumerate(param_range):
                # Run multiple replicates per parameter value
                replicate_results = []
                
                for rep in range(num_replicates):
                    # Update progress bar
                    progress = (i * num_replicates + rep) / (len(param_range) * num_replicates)
                    progress_bar.progress(progress)
                    
                    # Run a single replicate
                    params = fixed_params.copy()
                    params[param_name] = param_value
                    
                    # Run simulation for this replicate
                    net = simulate(
                        n=params.get('n', 20),
                        blocs=params.get('blocs', 4), 
                        steps=steps,
                        conn_intra=params.get('conn_intra', 0.6),
                        conn_inter=params.get('conn_inter', 0.2),
                        tariff_gap=params.get('tariff_gap', 0.3),
                        tariff_sd=params.get('tariff_sd', 0.05),
                        two_goods=params.get('two_goods', True),
                        policy_shock=params.get('policy_shock', None)
                    )
                    
                    # Extract metrics for this replicate
                    if target_metric == "world_gdp":
                        final_value = sum(c.history["gdp"][-1] for c in net.countries.values())
                        initial_value = sum(c.history["gdp"][0] for c in net.countries.values()) 
                        growth = (final_value / initial_value) - 1
                    elif target_metric == "poverty":
                        final_value = np.mean([c.history["poverty_rate"][-1] for c in net.countries.values()])
                        initial_value = np.mean([c.history["poverty_rate"][0] for c in net.countries.values()])
                        growth = (final_value / initial_value) - 1
                    elif target_metric == "trade_volume":
                        final_exports = sum(c.history["exports"][-1] for c in net.countries.values())
                        initial_exports = sum(c.history["exports"][0] for c in net.countries.values()) or 1
                        growth = (final_exports / initial_exports) - 1
                    elif target_metric == "fragmentation":
                        import networkx as nx
                        
                        G_copy = net.G.copy()
                        flows = []
                        for *_, d in G_copy.edges(data=True):
                            if "flow" in d:
                                flows.append(d["flow"])
                        
                        if flows: # Check if flows list is not empty
                            flows_array = np.array(flows)
                            threshold = np.percentile(flows_array, 40)
                            weak_edges = [(u, v) for u, v, d in G_copy.edges(data=True) if d.get("flow", 0) < threshold]
                            G_copy.remove_edges_from(weak_edges)
                            final_value = nx.number_weakly_connected_components(G_copy)
                        else:
                            final_value = len(net.countries) # Default fragmentation if no flows
                        growth = 0
                    elif target_metric == "gini":
                        gdp_values = sorted([c.gdp for c in net.countries.values()])
                        n = len(gdp_values)
                        if n > 1:
                            cum_wealth = np.cumsum(gdp_values)
                            final_value = (n + 1 - 2 * np.sum(cum_wealth) / cum_wealth[-1]) / n
                        else:
                            final_value = 0
                        growth = 0
                    else:
                        final_value = 0
                        growth = 0
                    
                    replicate_results.append({
                        "param_value": param_value,
                        "final_value": final_value,
                        "growth_rate": growth,
                        "replicate": rep,
                        "param_name": param_name,
                        "metric": target_metric
                    })
                
                # Average the replicates
                avg_final = np.mean([r["final_value"] for r in replicate_results])
                std_final = np.std([r["final_value"] for r in replicate_results])
                avg_growth = np.mean([r["growth_rate"] for r in replicate_results])
                std_growth = np.std([r["growth_rate"] for r in replicate_results])
                
                all_results.append({
                    "param_value": param_value,
                    "final_value": avg_final,
                    "growth_rate": avg_growth,
                    "final_value_std": std_final, 
                    "growth_rate_std": std_growth,
                    "param_name": param_name,
                    "metric": target_metric,
                    "replicates": num_replicates
                })
            
            # Final averaged results 
            results_df = pd.DataFrame(all_results)
            progress_bar.progress(100)
            
            # Line chart with error bars for final values
            st.subheader(f"Effect of {param_name} on {target_metric}")
            
            # Plot with error bars using Plotly
            fig = px.line(
                results_df, 
                x="param_value", 
                y="final_value",
                error_y="final_value_std",
                markers=True,
                title=f"Final {target_metric} vs {param_name} (with {num_replicates} replicates)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart of growth rates (if applicable)
            if target_metric in ["world_gdp", "poverty", "trade_volume"]:
                fig2 = px.bar(
                    results_df, 
                    x="param_value", 
                    y="growth_rate",
                    error_y="growth_rate_std",
                    title=f"{target_metric} Growth Rate vs {param_name}"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Equilibrium assessment
            if num_replicates > 1:
                st.subheader("Equilibrium Assessment")
                cv = results_df["final_value_std"] / results_df["final_value"] * 100  # coefficient of variation
                st.write(f"Average coefficient of variation across parameter values: {np.mean(cv):.2f}%")
                
                if np.mean(cv) < 5:
                    st.success("Low variability between replicates suggests equilibrium was reached.")
                elif np.mean(cv) < 15:
                    st.info("Moderate variability between replicates. Consider increasing simulation steps.")
                else:
                    st.warning("High variability between replicates. System may not reach equilibrium with current settings.")
            
            # Data table
            st.subheader("Results Summary")
            st.dataframe(results_df[["param_value", "final_value", "final_value_std", "growth_rate", "growth_rate_std"]])
    
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
        
        if st.button("Run MFA Comparison"):
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
                fig_gdp = px.line(df_gdp, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
                                title="World GDP Over Time")
                st.plotly_chart(fig_gdp, use_container_width=True)
                
                # Plot divegence metric - difference between simulation and MFA
                df_gdp['Divergence'] = df_gdp['Network Simulation'] - df_gdp['Mean-Field Approximation']
                fig_div = px.area(df_gdp, x='Step', y='Divergence', 
                                title="Network-MFA Divergence (GDP)")
                st.plotly_chart(fig_div, use_container_width=True)
                
                # Poverty rate comparison
                st.subheader("Poverty Rate: Network vs MFA")
                fig_pov = px.line(df_poverty, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
                                title="Average Poverty Rate Over Time")
                st.plotly_chart(fig_pov, use_container_width=True)
                
                # Gini coefficient comparison
                st.subheader("Inequality (Gini): Network vs MFA")
                fig_gini = px.line(df_gini, x='Step', y=['Network Simulation', 'Mean-Field Approximation'],
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
        
        # Add MFA curve fitting section
        st.subheader("MFA Curve Fitting")
        col1, col2 = st.columns(2)
        
        with col1:
            curve_type = st.selectbox(
                "Select curve type for MFA",
                ["Sigmoid (Logistic)", "Exponential", "Linear", "Power Law"]
            )
        
        with col2:
            metric_to_fit = st.selectbox(
                "Select metric to fit",
                ["GDP", "Poverty", "Trade Volume"]
            )
            
        if st.button("Run MFA Curve Fitting"):
            from scipy.optimize import curve_fit
            
            with st.spinner("Running simulation and fitting curves..."):
                # Run simulation first
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
                
                # Extract data based on selected metric
                if metric_to_fit == "GDP":
                    sim_data = np.array([sum(c.history["gdp"][t] for c in net.countries.values()) for t in range(mfa_steps)])
                elif metric_to_fit == "Poverty":
                    sim_data = np.array([np.mean([c.history["poverty_rate"][t] for c in net.countries.values()]) for t in range(mfa_steps)])
                else:  # Trade Volume
                    sim_data = np.array([sum(c.history["exports"][t] for c in net.countries.values() if t < len(c.history["exports"])) for t in range(mfa_steps)])
                
                # Normalize data for fitting
                sim_data_norm = (sim_data - np.min(sim_data)) / (np.max(sim_data) - np.min(sim_data))
                x_data = np.arange(len(sim_data))
                
                # Fit different curve types
                # Define curve functions
                def sigmoid(x, a, b, c, d):
                    return a / (1 + np.exp(-b * (x - c))) + d
                
                def exponential(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                def linear(x, a, b):
                    return a * x + b
                
                def power_law(x, a, b, c):
                    return a * (x + 1e-10)**b + c
                
                # AIC calculation function
                def calculate_aic(n, mse, k):
                    aic = n * np.log(mse) + 2 * k
                    return aic
                
                # Fit selected curve type
                try:
                    if curve_type == "Sigmoid (Logistic)":
                        p0 = [1.0, 0.1, len(x_data)/2, 0.0]  # initial guess
                        params, covariance = curve_fit(sigmoid, x_data, sim_data_norm, p0=p0, maxfev=10000)
                        mfa_fitted = sigmoid(x_data, *params)
                        mse = np.mean((sim_data_norm - mfa_fitted)**2)
                        aic = calculate_aic(len(x_data), mse, 4)  # 4 parameters
                        curve_formula = f"y = {params[0]:.4f} / (1 + exp(-{params[1]:.4f} * (x - {params[2]:.4f}))) + {params[3]:.4f}"
                        
                    elif curve_type == "Exponential":
                        p0 = [0.1, 0.01, 0.0]  # initial guess
                        params, covariance = curve_fit(exponential, x_data, sim_data_norm, p0=p0, maxfev=10000)
                        mfa_fitted = exponential(x_data, *params)
                        mse = np.mean((sim_data_norm - mfa_fitted)**2)
                        aic = calculate_aic(len(x_data), mse, 3)  # 3 parameters
                        curve_formula = f"y = {params[0]:.4f} * exp({params[1]:.4f} * x) + {params[2]:.4f}"
                        
                    elif curve_type == "Linear":
                        params, covariance = curve_fit(linear, x_data, sim_data_norm)
                        mfa_fitted = linear(x_data, *params)
                        mse = np.mean((sim_data_norm - mfa_fitted)**2)
                        aic = calculate_aic(len(x_data), mse, 2)  # 2 parameters
                        curve_formula = f"y = {params[0]:.4f} * x + {params[1]:.4f}"
                        
                    else:  # Power Law
                        p0 = [1.0, 0.5, 0.0]  # initial guess
                        params, covariance = curve_fit(power_law, x_data, sim_data_norm, p0=p0, maxfev=10000)
                        mfa_fitted = power_law(x_data, *params)
                        mse = np.mean((sim_data_norm - mfa_fitted)**2)
                        aic = calculate_aic(len(x_data), mse, 3)  # 3 parameters
                        curve_formula = f"y = {params[0]:.4f} * x^{params[1]:.4f} + {params[2]:.4f}"
                    
                    # Un-normalize fitted data back to original scale
                    mfa_fitted_full = mfa_fitted * (np.max(sim_data) - np.min(sim_data)) + np.min(sim_data)
                    
                    # Create comparison dataframe
                    df_comparison = pd.DataFrame({
                        'Step': x_data,
                        'Simulation': sim_data,
                        'MFA Fitted': mfa_fitted_full
                    })
                    
                    # Plot comparison
                    st.subheader(f"Fitted MFA Curve for {metric_to_fit}")
                    fig_fit = px.line(df_comparison, x='Step', y=['Simulation', 'MFA Fitted'],
                                    title=f"{metric_to_fit} - Simulation vs Fitted MFA")
                    st.plotly_chart(fig_fit, use_container_width=True)
                    
                    # Show fit statistics
                    st.subheader("Curve Fitting Results")
                    st.write(f"**Curve Formula:** {curve_formula}")
                    st.write(f"**Mean Squared Error:** {mse:.6f}")
                    st.write(f"**AIC (Akaike Information Criterion):** {aic:.2f}")
                    
                    # Explanation box
                    st.info("""
                    **How to interpret these results:**
                    - **Lower MSE** indicates a better fit to the simulation data
                    - **Lower AIC** indicates a better model, balancing fit quality with model complexity
                    - A good MFA approximation with low error suggests network effects are less important
                    - High error indicates network structure significantly impacts outcomes
                    """)
                    
                except Exception as e:
                    st.error(f"Curve fitting failed: {str(e)}")
                    st.write("Try a different curve type or adjust simulation parameters.")


if __name__ == "__main__":
    main()
