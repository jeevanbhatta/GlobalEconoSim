import numpy as np
import pandas as pd
import networkx as nx
from model import TradeNetwork

def get_top_gdp_countries(net: TradeNetwork, n=5):
    """Get the top n countries by GDP with detailed statistics"""
    # Find top n countries by GDP
    top_gdp_countries = sorted(net.countries.values(), key=lambda c: c.gdp, reverse=True)[:n]
    
    # Create a DataFrame for better display
    top_gdp_data = []
    for c in top_gdp_countries:
        # Get incoming and outgoing tariffs for this country
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(c.cid, data=True)]
        incoming_tariffs = [data['tariff'] for _, _, data in net.G.in_edges(c.cid, data=True)]
        
        # Calculate average tariffs if they exist
        avg_out_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        avg_in_tariff = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        # Calculate growth rate
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
        else:
            growth_rate = 0
            
        top_gdp_data.append({
            "Country ID": c.cid,
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Trade Balance": c.exports - c.imports,
            "Trade Balance Formatted": f"{c.exports - c.imports:,.0f}",
            "Avg Out Tariff": avg_out_tariff,
            "Avg Out Tariff Formatted": f"{avg_out_tariff:.2f}",
            "Avg In Tariff": avg_in_tariff,
            "Avg In Tariff Formatted": f"{avg_in_tariff:.2f}",
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%"
        })
    
    return pd.DataFrame(top_gdp_data)


def get_bottom_gdp_countries(net: TradeNetwork, n=5):
    """Get the bottom n countries by GDP with detailed statistics"""
    # Find bottom n countries by GDP
    bottom_gdp_countries = sorted(net.countries.values(), key=lambda c: c.gdp)[:n]
    
    # Create a DataFrame for better display
    bottom_gdp_data = []
    for c in bottom_gdp_countries:
        # Get incoming and outgoing tariffs for this country
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(c.cid, data=True)]
        incoming_tariffs = [data['tariff'] for _, _, data in net.G.in_edges(c.cid, data=True)]
        
        # Calculate average tariffs if they exist
        avg_out_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        avg_in_tariff = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        # Calculate growth rate
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
        else:
            growth_rate = 0
            
        bottom_gdp_data.append({
            "Country ID": c.cid,
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Trade Balance": c.exports - c.imports,
            "Trade Balance Formatted": f"{c.exports - c.imports:,.0f}",
            "Avg Out Tariff": avg_out_tariff,
            "Avg Out Tariff Formatted": f"{avg_out_tariff:.2f}",
            "Avg In Tariff": avg_in_tariff,
            "Avg In Tariff Formatted": f"{avg_in_tariff:.2f}",
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%"
        })
    
    return pd.DataFrame(bottom_gdp_data)


def get_top_connected_countries(net: TradeNetwork, n=5):
    """Get the top n countries by number of trade connections"""
    # Count connections per country
    country_connections = {}
    for cid in net.G.nodes():
        country_connections[cid] = net.G.degree(cid)
    
    top_connected_cids = sorted(country_connections.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Create a DataFrame for display
    top_conn_data = []
    for cid, degree in top_connected_cids:
        c = net.countries[cid]
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(cid, data=True)]
        incoming_tariffs = [data['tariff'] for _, _, data in net.G.in_edges(cid, data=True)]
        
        avg_out_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        avg_in_tariff = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        # Calculate growth rate
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
        else:
            growth_rate = 0
            
        top_conn_data.append({
            "Country ID": cid,
            "Connections": degree,
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Avg Out Tariff": avg_out_tariff,
            "Avg Out Tariff Formatted": f"{avg_out_tariff:.2f}",
            "Avg In Tariff": avg_in_tariff,
            "Avg In Tariff Formatted": f"{avg_in_tariff:.2f}",
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%"
        })
    
    return pd.DataFrame(top_conn_data)


def get_least_connected_countries(net: TradeNetwork, n=5):
    """Get the least n countries by number of trade connections"""
    # Count connections per country
    country_connections = {}
    for cid in net.G.nodes():
        country_connections[cid] = net.G.degree(cid)
    
    least_connected_cids = sorted(country_connections.items(), key=lambda x: x[1])[:n]
    
    # Create a DataFrame for display
    least_conn_data = []
    for cid, degree in least_connected_cids:
        c = net.countries[cid]
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(cid, data=True)]
        incoming_tariffs = [data['tariff'] for _, _, data in net.G.in_edges(cid, data=True)]
        
        avg_out_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        avg_in_tariff = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        # Calculate growth rate
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
        else:
            growth_rate = 0
            
        least_conn_data.append({
            "Country ID": cid,
            "Connections": degree,
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Avg Out Tariff": avg_out_tariff,
            "Avg Out Tariff Formatted": f"{avg_out_tariff:.2f}",
            "Avg In Tariff": avg_in_tariff,
            "Avg In Tariff Formatted": f"{avg_in_tariff:.2f}",
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%"
        })
    
    return pd.DataFrame(least_conn_data)


def get_trade_efficiency_countries(net: TradeNetwork, n=5, reverse=False):
    """Get countries with highest/lowest trade efficiency (Trade Volume / GDP)"""
    trade_efficiency = []
    for cid, c in net.countries.items():
        if c.gdp > 0:  # Avoid division by zero
            efficiency = (c.exports + c.imports) / c.gdp
            trade_efficiency.append((cid, efficiency))
    
    if not trade_efficiency:
        return pd.DataFrame()  # Return empty DataFrame if no data
        
    # Sort by efficiency (highest or lowest)
    if not reverse:
        efficient_countries = sorted(trade_efficiency, key=lambda x: x[1], reverse=True)[:n]
    else:
        efficient_countries = sorted(trade_efficiency, key=lambda x: x[1])[:n]
    
    # Create a DataFrame for display
    eff_data = []
    for cid, efficiency in efficient_countries:
        c = net.countries[cid]
        
        # Calculate growth rate
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
        else:
            growth_rate = 0
            
        eff_data.append({
            "Country ID": cid,
            "Trade Efficiency": efficiency,
            "Trade Efficiency Formatted": f"{efficiency:.3f}",
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%"
        })
    
    return pd.DataFrame(eff_data)


def get_growth_rate_countries(net: TradeNetwork, n=5, highest=True):
    """Get countries with highest/lowest GDP growth rate"""
    growth_rates = []
    for cid, c in net.countries.items():
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
            growth_rates.append((cid, growth_rate))
    
    if not growth_rates:
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    # Sort by growth rate (highest or lowest)
    if highest:
        sorted_countries = sorted(growth_rates, key=lambda x: x[1], reverse=True)[:n]
    else:
        sorted_countries = sorted(growth_rates, key=lambda x: x[1])[:n]
    
    # Create a DataFrame for display
    growth_data = []
    for cid, growth_rate in sorted_countries:
        c = net.countries[cid]
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(cid, data=True)]
        incoming_tariffs = [data['tariff'] for _, _, data in net.G.in_edges(cid, data=True)]
        
        avg_out_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        avg_in_tariff = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        connections = net.G.degree(cid)
        
        growth_data.append({
            "Country ID": cid,
            "Growth Rate": growth_rate,
            "Growth Rate Formatted": f"{growth_rate*100:.1f}%",
            "GDP": c.gdp,
            "GDP Formatted": f"{c.gdp:,.0f}",
            "Bloc": c.bloc,
            "Connections": connections,
            "Exports": c.exports,
            "Exports Formatted": f"{c.exports:,.0f}",
            "Imports": c.imports,
            "Imports Formatted": f"{c.imports:,.0f}",
            "Avg Out Tariff": avg_out_tariff,
            "Avg Out Tariff Formatted": f"{avg_out_tariff:.2f}",
            "Avg In Tariff": avg_in_tariff,
            "Avg In Tariff Formatted": f"{avg_in_tariff:.2f}"
        })
    
    return pd.DataFrame(growth_data)


def calculate_gdp_tariff_correlation(net: TradeNetwork):
    """Calculate correlation between GDP and average tariffs"""
    all_gdps = [c.gdp for c in net.countries.values()]
    all_avg_tariffs = []
    
    for cid in net.countries:
        country_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(cid, data=True)]
        avg_tariff = sum(country_tariffs) / len(country_tariffs) if country_tariffs else 0
        all_avg_tariffs.append(avg_tariff)
    
    # Calculate correlation coefficient if we have enough data
    if len(all_gdps) > 1 and len(all_avg_tariffs) > 1:
        correlation = np.corrcoef(all_gdps, all_avg_tariffs)[0, 1]
        return correlation, all_gdps, all_avg_tariffs
    else:
        return 0, all_gdps, all_avg_tariffs


def calculate_gdp_connections_correlation(net: TradeNetwork):
    """Calculate correlation between GDP and number of connections"""
    all_gdps = [c.gdp for c in net.countries.values()]
    all_connections = [net.G.degree(cid) for cid in net.countries]
    
    # Calculate correlation coefficient if we have enough data
    if len(all_gdps) > 1 and len(all_connections) > 1:
        correlation = np.corrcoef(all_gdps, all_connections)[0, 1]
        return correlation, all_gdps, all_connections
    else:
        return 0, all_gdps, all_connections


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
    
    # Initialize neighbor variables with default values
    neighbor_initial_gdp = 0
    neighbor_final_gdp = 0
    neighbor_growth = 0
    neighbor_poverty = 0
    
    # Calculate metrics for neighbors
    if neighbors:
        neighbor_initial_gdp = np.mean([network.countries[n].history['gdp'][0] for n in neighbors])
        neighbor_final_gdp = np.mean([network.countries[n].history['gdp'][-1] for n in neighbors])
        neighbor_growth = (neighbor_final_gdp / neighbor_initial_gdp - 1) if neighbor_initial_gdp > 0 else 0
        neighbor_poverty = np.mean([network.countries[n].poverty_rate for n in neighbors])
    
    # Calculate metrics for all countries
    all_countries = list(network.countries.values())
    all_initial_gdp = np.mean([c.history['gdp'][0] for c in all_countries])
    all_final_gdp = np.mean([c.history['gdp'][-1] for c in all_countries])
    all_growth = (all_final_gdp / all_initial_gdp - 1) if all_initial_gdp > 0 else 0
    all_poverty = np.mean([c.poverty_rate for c in all_countries])
    
    # Create comparison DataFrame
    data = {
        'Growth Rate': [shocked_growth, neighbor_growth, all_growth],
        'Final GDP': [shocked_final_gdp, neighbor_final_gdp, all_final_gdp],
        'Initial GDP': [shocked_initial_gdp, neighbor_initial_gdp, all_initial_gdp],
        'Poverty Rate': [
            shocked_country.poverty_rate, 
            neighbor_poverty,
            all_poverty
        ]
    }
    
    df = pd.DataFrame(data, index=['Shocked Country', 'Neighbors', 'World Average'])
    return df