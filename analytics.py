import numpy as np
import pandas as pd
import networkx as nx  # Added missing import
from model import TradeNetwork, simulate  # Import simulate at the module level

def mfa_series(net: TradeNetwork, steps: int):
    n = len(net.countries)
    avg_growth = np.mean([
        (c.history["gdp"][-1] / c.history["gdp"][0]) ** (1 / steps) - 1 for c in net.countries.values()
    ])
    gdp0 = np.mean([c.history["gdp"][0] for c in net.countries.values()])
    world = []
    for t in range(steps):
        gdp_t = gdp0 * ((1 + avg_growth) ** t)
        world.append(gdp_t * n)
    return world

def parameter_sensitivity_analysis(param_name, param_range, fixed_params, target_metric="world_gdp", steps=60, num_replicates=3):
    results = []
    for param_value in param_range:
        replicate_results = []
        for rep in range(num_replicates):
            # Initialize variables with default values to avoid UnboundLocalError
            final_value = 0
            initial_value = 0
            growth = 0
            
            try:
                params = fixed_params.copy()
                params[param_name] = param_value
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
                    G_copy = net.G.copy()
                    flows = [d["flow"] for *_, d in G_copy.edges(data=True) if "flow" in d]
                    if flows:
                        flows_array = np.array(flows)
                        threshold = np.percentile(flows_array, 40)
                        weak_edges = [(u, v) for u, v, d in G_copy.edges(data=True) if d.get("flow", 0) < threshold]
                        G_copy.remove_edges_from(weak_edges)
                        final_value = nx.number_weakly_connected_components(G_copy)
                    else:
                        final_value = len(net.countries)
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
            except Exception as e:
                import traceback
                print(f"Error in replicate {rep} for {param_name}={param_value}: {str(e)}")
                print(traceback.format_exc())
                # Keep the default values for final_value and growth
            
            replicate_results.append({
                "param_value": param_value,
                "final_value": final_value,
                "growth_rate": growth,
                "replicate": rep,
                "param_name": param_name,
                "metric": target_metric
            })
        avg_final = np.mean([r["final_value"] for r in replicate_results])
        std_final = np.std([r["final_value"] for r in replicate_results])
        avg_growth = np.mean([r["growth_rate"] for r in replicate_results])
        std_growth = np.std([r["growth_rate"] for r in replicate_results])
        results.append({
            "param_value": param_value,
            "final_value": avg_final,
            "growth_rate": avg_growth,
            "final_value_std": std_final,
            "growth_rate_std": std_growth,
            "param_name": param_name,
            "metric": target_metric,
            "replicates": num_replicates
        })
    return pd.DataFrame(results)
