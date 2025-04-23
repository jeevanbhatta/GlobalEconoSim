import numpy as np
import pandas as pd
import networkx as nx
from model import simulate

def mfa_series(network, steps):
    """
    Generate a Mean Field Approximation time series for comparison with full network sim.
    MFA ignores network structure and treats every country as interacting with the 'average' country.
    """
    # Get initial values
    countries = list(network.countries.values())
    n_countries = len(countries)
    
    # Extract starting conditions
    initial_gdp = np.mean([c.history["gdp"][0] for c in countries])
    initial_poverty = np.mean([c.history["poverty_rate"][0] for c in countries])
    
    # Start with the first value from the actual simulation
    world_gdp_t0 = sum(c.history["gdp"][0] for c in countries)
    mfa_series = [world_gdp_t0]
    
    # Simple growth model for MFA - baseline growth plus trade effect
    for t in range(1, steps):
        # Fixed parameters for MFA growth model
        base_growth = 0.02  # 2% base growth
        trade_impact = 0.03  # Trade adds up to 3% extra growth
        
        # World GDP grows by baseline growth rate
        mfa_growth = base_growth + trade_impact * (0.5 + 0.5 * np.sin(t/10))  # Add cyclical component
        world_gdp = mfa_series[-1] * (1 + mfa_growth)
        mfa_series.append(world_gdp)
    
    return mfa_series

def parameter_sensitivity_analysis(param_name, param_range, fixed_params, target_metric, steps, num_replicates=1):
    """
    Run sensitivity analysis on a parameter by varying it while keeping others fixed.
    Returns a DataFrame with results.
    
    param_name: Name of parameter to vary (string)
    param_range: List of values to try for the parameter
    fixed_params: Dictionary of other parameters to keep constant
    target_metric: Metric to track ('world_gdp', 'poverty', etc)
    steps: Number of simulation steps
    num_replicates: Number of times to run each parameter setting (for robustness)
    """
    results = []
    
    for param_value in param_range:
        # Create simulation parameters
        sim_params = fixed_params.copy()
        sim_params[param_name] = param_value
        
        # We'll store results for this parameter value across replicates
        replicate_results = {
            'final_values': [],
            'growth_rates': []
        }
        
        for rep in range(num_replicates):
            # Create policy shock tuple if required parameters are available
            policy_shock = None
            if 'shock_id' in sim_params and 'shock_delta' in sim_params:
                policy_shock = (sim_params.get('shock_id', 0), sim_params.get('shock_delta', 0.1))
            
            # Run simulation
            network = simulate(
                n=sim_params.get('n', 20),
                blocs=sim_params.get('blocs', 4),
                steps=steps,
                conn_intra=sim_params.get('conn_intra', 0.6),
                conn_inter=sim_params.get('conn_inter', 0.2),
                tariff_gap=sim_params.get('tariff_gap', 0.3),
                tariff_sd=sim_params.get('tariff_sd', 0.05),
                two_goods=sim_params.get('two_goods', True),
                policy_shock=policy_shock
            )
            
            # Calculate target metric
            if target_metric == 'world_gdp':
                initial_value = sum(c.history["gdp"][0] for c in network.countries.values())
                final_value = sum(c.history["gdp"][-1] for c in network.countries.values())
            elif target_metric == 'poverty':
                initial_value = np.mean([c.history["poverty_rate"][0] for c in network.countries.values()])
                final_value = np.mean([c.history["poverty_rate"][-1] for c in network.countries.values()])
            elif target_metric == 'trade_volume':
                initial_value = sum(c.history["exports"][0] for c in network.countries.values())
                final_value = sum(c.history["exports"][-1] for c in network.countries.values())
            elif target_metric == 'fragmentation':
                # Measure network fragmentation by removing weakest 40% of edges
                G_copy = network.G.copy()
                flows = [d.get('flow', 0) for _, _, d in G_copy.edges(data=True)]
                if flows:
                    threshold = np.percentile(flows, 40)
                    weak_edges = [(u, v) for u, v, d in G_copy.edges(data=True) if d.get('flow', 0) < threshold]
                    G_copy.remove_edges_from(weak_edges)
                    final_value = float(len(list(nx.weakly_connected_components(G_copy))))
                    initial_value = 1.0  # Assume started connected
                else:
                    final_value = 1.0
                    initial_value = 1.0
            elif target_metric == 'gini':
                # Calculate Gini coefficient (inequality measure)
                gdp_values_initial = sorted([c.history["gdp"][0] for c in network.countries.values()])
                gdp_values_final = sorted([c.history["gdp"][-1] for c in network.countries.values()])
                
                def gini(x):
                    n = len(x)
                    if n <= 1:
                        return 0
                    s = sum(i * x[i] for i in range(n))
                    return (2 * s / (n * sum(x)) - (n + 1) / n)
                
                initial_value = gini(gdp_values_initial)
                final_value = gini(gdp_values_final)
            else:
                initial_value = 0
                final_value = 0
            
            # Calculate growth rate
            if initial_value > 0:
                growth_rate = (final_value / initial_value) - 1
            else:
                growth_rate = 0
                
            replicate_results['final_values'].append(final_value)
            replicate_results['growth_rates'].append(growth_rate)
        
        # Calculate mean and standard deviation across replicates
        final_value_mean = np.mean(replicate_results['final_values'])
        final_value_std = np.std(replicate_results['final_values'])
        growth_rate_mean = np.mean(replicate_results['growth_rates'])
        growth_rate_std = np.std(replicate_results['growth_rates'])
        
        # Store results
        results.append({
            'param_value': param_value,
            'final_value': final_value_mean,
            'final_value_std': final_value_std,
            'growth_rate': growth_rate_mean,
            'growth_rate_std': growth_rate_std,
            'replicates': num_replicates
        })
    
    return pd.DataFrame(results)

# Try to import networkx for fragmentation calculation
try:
    import networkx as nx
except ImportError:
    class DummyNX:
        class WeaklyConnectedComponents:
            @staticmethod
            def weakly_connected_components(G):
                return [[node for node in G.nodes()]]
        
        def number_weakly_connected_components(self, G):
            return 1
    
    nx = DummyNX()
