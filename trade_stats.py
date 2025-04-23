import numpy as np
import pandas as pd
import networkx as nx

def get_top_gdp_countries(network, n=5):
    """Returns the top n countries by GDP"""
    countries = list(network.countries.values())
    top_countries = sorted(countries, key=lambda c: c.gdp, reverse=True)[:n]
    
    # Create DataFrame for display
    data = []
    for c in top_countries:
        data.append({
            'Country ID': c.cid,
            'Bloc': c.bloc,
            'GDP': c.gdp,
            'Population': c.population,
            'GDP per capita': c.gdp / c.population * 1000,  # Scale for readability
            'Poverty Rate': c.poverty_rate
        })
    return pd.DataFrame(data)

def get_bottom_gdp_countries(network, n=5):
    """Returns the bottom n countries by GDP"""
    countries = list(network.countries.values())
    bottom_countries = sorted(countries, key=lambda c: c.gdp)[:n]
    
    # Create DataFrame for display
    data = []
    for c in bottom_countries:
        data.append({
            'Country ID': c.cid,
            'Bloc': c.bloc,
            'GDP': c.gdp,
            'Population': c.population,
            'GDP per capita': c.gdp / c.population * 1000,  # Scale for readability
            'Poverty Rate': c.poverty_rate
        })
    return pd.DataFrame(data)

def get_top_connected_countries(network, n=5):
    """Returns the top n countries by number of trading connections"""
    # Calculate degree for each country
    degree_dict = dict(network.G.degree())
    top_countries = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Create DataFrame for display
    data = []
    for cid, degree in top_countries:
        c = network.countries[cid]
        data.append({
            'Country ID': c.cid,
            'Bloc': c.bloc,
            'Trade Connections': degree,
            'GDP': c.gdp,
            'Exports': c.exports,
            'Imports': c.imports
        })
    return pd.DataFrame(data)

def get_least_connected_countries(network, n=5):
    """Returns the least n countries by number of trading connections"""
    # Calculate degree for each country
    degree_dict = dict(network.G.degree())
    bottom_countries = sorted(degree_dict.items(), key=lambda x: x[1])[:n]
    
    # Create DataFrame for display
    data = []
    for cid, degree in bottom_countries:
        c = network.countries[cid]
        data.append({
            'Country ID': c.cid,
            'Bloc': c.bloc,
            'Trade Connections': degree,
            'GDP': c.gdp,
            'Exports': c.exports,
            'Imports': c.imports
        })
    return pd.DataFrame(data)

def get_trade_efficiency_countries(network, n=5, top=True):
    """Returns countries sorted by trade efficiency (exports per connection)"""
    data = []
    for cid, country in network.countries.items():
        # Get outgoing connections
        connections = list(network.G.successors(cid))
        num_connections = len(connections)
        
        if num_connections > 0:
            efficiency = country.exports / max(1, num_connections)  # Avoid division by zero
        else:
            efficiency = 0
            
        data.append({
            'Country ID': cid,
            'Bloc': country.bloc,
            'Exports': country.exports,
            'Connections': num_connections,
            'Efficiency': efficiency,
            'GDP': country.gdp
        })
    
    df = pd.DataFrame(data)
    if top:
        # Return top n by efficiency
        return df.sort_values('Efficiency', ascending=False).head(n)
    else:
        # Return bottom n by efficiency
        return df.sort_values('Efficiency').head(n)

def get_growth_rate_countries(network, n=5, top=True):
    """Returns countries with highest/lowest GDP growth rates"""
    data = []
    for cid, country in network.countries.items():
        if len(country.history['gdp']) >= 2:
            initial_gdp = country.history['gdp'][0]
            final_gdp = country.history['gdp'][-1]
            
            # Calculate average annualized growth rate
            num_periods = len(country.history['gdp']) - 1
            if initial_gdp > 0 and num_periods > 0:
                growth_rate = (final_gdp / initial_gdp) ** (1 / num_periods) - 1
            else:
                growth_rate = 0
                
            data.append({
                'Country ID': cid,
                'Bloc': country.bloc,
                'Initial GDP': initial_gdp,
                'Final GDP': final_gdp,
                'Growth Rate': growth_rate,
                'Poverty Rate': country.poverty_rate
            })
    
    df = pd.DataFrame(data)
    if top:
        # Return top n by growth rate
        return df.sort_values('Growth Rate', ascending=False).head(n)
    else:
        # Return bottom n by growth rate
        return df.sort_values('Growth Rate').head(n)

def calculate_gdp_tariff_correlation(network):
    """Calculate correlation between average tariffs and GDP"""
    countries = list(network.countries.values())
    
    # Calculate average tariff for each country's outgoing edges
    avg_tariffs = []
    gdp_values = []
    
    for cid, country in network.countries.items():
        outgoing_edges = list(network.G.out_edges(cid, data=True))
        if outgoing_edges:
            avg_tariff = np.mean([d['tariff'] for _, _, d in outgoing_edges])
            avg_tariffs.append(avg_tariff)
            gdp_values.append(country.gdp)
    
    # Calculate correlation
    if len(avg_tariffs) > 1:
        correlation = np.corrcoef(avg_tariffs, gdp_values)[0, 1]
    else:
        correlation = 0
        
    return correlation, avg_tariffs, gdp_values

def calculate_gdp_connections_correlation(network):
    """Calculate correlation between number of trade connections and GDP"""
    degree_dict = dict(network.G.degree())
    
    connections = []
    gdp_values = []
    
    for cid, degree in degree_dict.items():
        connections.append(degree)
        gdp_values.append(network.countries[cid].gdp)
    
    # Calculate correlation
    if len(connections) > 1:
        correlation = np.corrcoef(connections, gdp_values)[0, 1]
    else:
        correlation = 0
        
    return correlation, connections, gdp_values

def get_shock_country_comparison(network, shock_id):
    """
    Compare a shocked country to its neighbors and the world average
    Returns a DataFrame with comparison metrics
    """
    if shock_id not in network.countries:
        return None
    
    shocked_country = network.countries[shock_id]
    
    # Get neighbors of shocked country
    neighbors = list(network.G.successors(shock_id)) + list(network.G.predecessors(shock_id))
    neighbors = list(set(neighbors))  # Remove duplicates
    
    # Calculate metrics for shocked country
    shocked_initial_gdp = shocked_country.history['gdp'][0]
    shocked_final_gdp = shocked_country.history['gdp'][-1]
    shocked_growth = (shocked_final_gdp / shocked_initial_gdp - 1) if shocked_initial_gdp > 0 else 0
    
    # Calculate metrics for neighbors
    if neighbors:
        neighbor_initial_gdp = np.mean([network.countries[n].history['gdp'][0] for n in neighbors])
        neighbor_final_gdp = np.mean([network.countries[n].history['gdp'][-1] for n in neighbors])
        neighbor_growth = (neighbor_final_gdp / neighbor_initial_gdp - 1) if neighbor_initial_gdp > 0 else 0
    else:
        neighbor_growth = 0
    
    # Calculate metrics for all countries
    all_countries = list(network.countries.values())
    all_initial_gdp = np.mean([c.history['gdp'][0] for c in all_countries])
    all_final_gdp = np.mean([c.history['gdp'][-1] for c in all_countries])
    all_growth = (all_final_gdp / all_initial_gdp - 1) if all_initial_gdp > 0 else 0
    
    # Create comparison DataFrame
    data = {
        'Growth Rate': [shocked_growth, neighbor_growth, all_growth],
        'Final GDP': [shocked_final_gdp, neighbor_final_gdp, all_final_gdp],
        'Initial GDP': [shocked_initial_gdp, neighbor_initial_gdp, all_initial_gdp],
        'Poverty Rate': [
            shocked_country.poverty_rate, 
            np.mean([network.countries[n].poverty_rate for n in neighbors]) if neighbors else 0,
            np.mean([c.poverty_rate for c in all_countries])
        ]
    }
    
    df = pd.DataFrame(data, index=['Shocked Country', 'Neighbors', 'World Average'])
    return df

def calculate_trade_balance_distribution(network):
    """Calculate trade balance (exports - imports) for each country"""
    balances = []
    for cid, country in network.countries.items():
        balance = country.exports - country.imports
        balances.append({
            'Country ID': cid,
            'Bloc': country.bloc,
            'Exports': country.exports,
            'Imports': country.imports,
            'Trade Balance': balance,
            'GDP': country.gdp,
            'Balance to GDP Ratio': balance / country.gdp if country.gdp > 0 else 0
        })
    
    return pd.DataFrame(balances)

def compare_bloc_performance(network):
    """Compare economic performance between different political blocs"""
    blocs = {}
    for cid, country in network.countries.items():
        bloc = country.bloc
        if bloc not in blocs:
            blocs[bloc] = {
                'GDP': 0,
                'Population': 0,
                'Countries': 0,
                'Avg Poverty': 0,
                'Total Exports': 0,
                'Total Imports': 0
            }
        
        blocs[bloc]['GDP'] += country.gdp
        blocs[bloc]['Population'] += country.population
        blocs[bloc]['Countries'] += 1
        blocs[bloc]['Avg Poverty'] += country.poverty_rate
        blocs[bloc]['Total Exports'] += country.exports
        blocs[bloc]['Total Imports'] += country.imports
    
    # Calculate averages
    for bloc in blocs:
        if blocs[bloc]['Countries'] > 0:
            blocs[bloc]['Avg Poverty'] /= blocs[bloc]['Countries']
            blocs[bloc]['GDP per Capita'] = (blocs[bloc]['GDP'] / blocs[bloc]['Population']) * 1000
            blocs[bloc]['Exports per Country'] = blocs[bloc]['Total Exports'] / blocs[bloc]['Countries']
            blocs[bloc]['Trade Balance'] = blocs[bloc]['Total Exports'] - blocs[bloc]['Total Imports']
    
    return pd.DataFrame(blocs).T

def get_growth_trajectory(network, country_id, normalize=True):
    """Returns the GDP growth trajectory for a specific country"""
    if country_id not in network.countries:
        return None
    
    country = network.countries[country_id]
    gdp_series = country.history['gdp']
    
    if normalize and len(gdp_series) > 0:
        initial_gdp = gdp_series[0]
        normalized_series = [gdp / initial_gdp for gdp in gdp_series]
        return pd.DataFrame({
            'Time Step': range(len(gdp_series)),
            'Normalized GDP': normalized_series,
            'Raw GDP': gdp_series
        })
    else:
        return pd.DataFrame({
            'Time Step': range(len(gdp_series)),
            'GDP': gdp_series
        })

def analyze_shock_effects(network, shock_id, before_window=5, after_window=15):
    """Analyze the effects of a shock on various metrics before and after the shock"""
    if 'friendship_stats_history' not in dir(network) or shock_id not in network.friendship_stats_history['shock_countries']:
        return None
    
    # Find the shock point for this country
    shock_point = None
    for t, sid, _ in network.friendship_stats_history['shock_points']:
        if sid == shock_id:
            shock_point = t
            break
    
    if shock_point is None:
        return None
    
    # Get data before and after shock
    country = network.countries[shock_id]
    time_points = network.friendship_stats_history['time']
    
    # Find valid indices for before and after windows
    start_before = max(0, shock_point - before_window)
    end_after = min(len(time_points), shock_point + after_window)
    
    # Get metrics for all available time points
    metrics = {
        'GDP': country.history['gdp'],
        'Poverty Rate': country.history['poverty_rate'],
        'Exports': country.history['exports'],
        'Imports': country.history['imports'],
        'Outgoing Tariffs': network.friendship_stats_history['shock_countries'][shock_id]['outgoing_avg_tariff'],
        'Incoming Tariffs': network.friendship_stats_history['shock_countries'][shock_id]['incoming_avg_tariff'],
        'Trading Partners': network.friendship_stats_history['shock_countries'][shock_id]['trading_partners']
    }
    
    # Calculate average values before and after shock
    before_after = {'Metric': [], 'Before Shock': [], 'After Shock': [], 'Change (%)': []}
    
    for metric_name, values in metrics.items():
        if len(values) <= shock_point:
            continue
            
        before_vals = values[start_before:shock_point]
        after_vals = values[shock_point:end_after]
        
        before_avg = sum(before_vals) / len(before_vals) if before_vals else 0
        after_avg = sum(after_vals) / len(after_vals) if after_vals else 0
        
        pct_change = ((after_avg / before_avg) - 1) * 100 if before_avg != 0 else 0
        
        before_after['Metric'].append(metric_name)
        before_after['Before Shock'].append(before_avg)
        before_after['After Shock'].append(after_avg)
        before_after['Change (%)'].append(pct_change)
    
    return pd.DataFrame(before_after)