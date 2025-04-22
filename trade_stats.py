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
        return np.corrcoef(all_gdps, all_avg_tariffs)[0, 1]
    else:
        return 0


def calculate_gdp_connections_correlation(net: TradeNetwork):
    """Calculate correlation between GDP and number of connections"""
    all_gdps = [c.gdp for c in net.countries.values()]
    all_connections = [net.G.degree(cid) for cid in net.countries]
    
    # Calculate correlation coefficient if we have enough data
    if len(all_gdps) > 1 and len(all_connections) > 1:
        return np.corrcoef(all_gdps, all_connections)[0, 1]
    else:
        return 0


def get_shock_country_comparison(net: TradeNetwork, shock_id: int):
    """
    Compares statistics of the shock country to mean and median values
    
    Args:
        net: The trade network
        shock_id: The ID of the country that received the policy shock
        
    Returns:
        DataFrame with shock country stats and comparison to mean/median values
    """
    if shock_id not in net.countries:
        return pd.DataFrame()
        
    # Get the shock country
    shock_country = net.countries[shock_id]
    
    # Calculate mean and median values across all countries
    all_countries = list(net.countries.values())
    
    # GDP stats
    all_gdps = [c.gdp for c in all_countries]
    mean_gdp = np.mean(all_gdps)
    median_gdp = np.median(all_gdps)
    
    # Export stats
    all_exports = [c.exports for c in all_countries]
    mean_exports = np.mean(all_exports)
    median_exports = np.median(all_exports)
    
    # Import stats
    all_imports = [c.imports for c in all_countries]
    mean_imports = np.mean(all_imports)
    median_imports = np.median(all_imports)
    
    # Trade balance stats
    all_trade_balances = [c.exports - c.imports for c in all_countries]
    mean_trade_balance = np.mean(all_trade_balances)
    median_trade_balance = np.median(all_trade_balances)
    
    # Trade efficiency stats
    all_efficiencies = [(c.exports + c.imports) / c.gdp if c.gdp > 0 else 0 for c in all_countries]
    mean_efficiency = np.mean(all_efficiencies)
    median_efficiency = np.median(all_efficiencies)
    
    # Growth rate stats
    all_growth_rates = []
    for c in all_countries:
        gdp_history = c.history["gdp"]
        if len(gdp_history) > 1 and gdp_history[0] > 0:
            growth_rate = (gdp_history[-1] / gdp_history[0]) - 1
            all_growth_rates.append(growth_rate)
        else:
            all_growth_rates.append(0)
    
    mean_growth = np.mean(all_growth_rates)
    median_growth = np.median(all_growth_rates)
    
    # Connections stats
    all_connections = [net.G.degree(c.cid) for c in all_countries]
    mean_connections = np.mean(all_connections)
    median_connections = np.median(all_connections)
    
    # Tariff stats
    country_avg_tariffs = []
    for c in all_countries:
        outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(c.cid, data=True)]
        avg_tariff = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        country_avg_tariffs.append(avg_tariff)
    
    mean_tariff = np.mean(country_avg_tariffs)
    median_tariff = np.median(country_avg_tariffs)
    
    # Get shock country's values
    shock_outgoing_tariffs = [data['tariff'] for _, _, data in net.G.out_edges(shock_id, data=True)]
    shock_avg_tariff = sum(shock_outgoing_tariffs) / len(shock_outgoing_tariffs) if shock_outgoing_tariffs else 0
    
    shock_connections = net.G.degree(shock_id)
    
    shock_gdp_history = shock_country.history["gdp"]
    if len(shock_gdp_history) > 1 and shock_gdp_history[0] > 0:
        shock_growth_rate = (shock_gdp_history[-1] / shock_gdp_history[0]) - 1
    else:
        shock_growth_rate = 0
    
    shock_trade_balance = shock_country.exports - shock_country.imports
    
    # Calculate trade efficiency for shock country
    if shock_country.gdp > 0:
        shock_efficiency = (shock_country.exports + shock_country.imports) / shock_country.gdp
    else:
        shock_efficiency = 0
    
    # Create comparison dataframe
    comparison_data = [
        {
            "Metric": "GDP",
            "Shock Country": shock_country.gdp,
            "Shock Value": f"{shock_country.gdp:,.0f}",
            "Mean": mean_gdp,
            "Mean Value": f"{mean_gdp:,.0f}",
            "Median": median_gdp,
            "Median Value": f"{median_gdp:,.0f}",
            "% Diff from Mean": (shock_country.gdp / mean_gdp - 1) * 100 if mean_gdp > 0 else 0,
            "% Diff Value": f"{(shock_country.gdp / mean_gdp - 1) * 100:.1f}%" if mean_gdp > 0 else "N/A"
        },
        {
            "Metric": "Exports",
            "Shock Country": shock_country.exports,
            "Shock Value": f"{shock_country.exports:,.0f}",
            "Mean": mean_exports,
            "Mean Value": f"{mean_exports:,.0f}",
            "Median": median_exports,
            "Median Value": f"{median_exports:,.0f}",
            "% Diff from Mean": (shock_country.exports / mean_exports - 1) * 100 if mean_exports > 0 else 0,
            "% Diff Value": f"{(shock_country.exports / mean_exports - 1) * 100:.1f}%" if mean_exports > 0 else "N/A"
        },
        {
            "Metric": "Imports",
            "Shock Country": shock_country.imports,
            "Shock Value": f"{shock_country.imports:,.0f}",
            "Mean": mean_imports,
            "Mean Value": f"{mean_imports:,.0f}",
            "Median": median_imports,
            "Median Value": f"{median_imports:,.0f}",
            "% Diff from Mean": (shock_country.imports / mean_imports - 1) * 100 if mean_imports > 0 else 0,
            "% Diff Value": f"{(shock_country.imports / mean_imports - 1) * 100:.1f}%" if mean_imports > 0 else "N/A"
        },
        {
            "Metric": "Trade Balance",
            "Shock Country": shock_trade_balance,
            "Shock Value": f"{shock_trade_balance:,.0f}",
            "Mean": mean_trade_balance,
            "Mean Value": f"{mean_trade_balance:,.0f}",
            "Median": median_trade_balance,
            "Median Value": f"{median_trade_balance:,.0f}",
            "% Diff from Mean": (shock_trade_balance / mean_trade_balance - 1) * 100 if mean_trade_balance != 0 else 0,
            "% Diff Value": f"{(shock_trade_balance / mean_trade_balance - 1) * 100:.1f}%" if mean_trade_balance != 0 else "N/A"
        },
        {
            "Metric": "Trade Efficiency",
            "Shock Country": shock_efficiency,
            "Shock Value": f"{shock_efficiency:.3f}",
            "Mean": mean_efficiency,
            "Mean Value": f"{mean_efficiency:.3f}",
            "Median": median_efficiency,
            "Median Value": f"{median_efficiency:.3f}",
            "% Diff from Mean": (shock_efficiency / mean_efficiency - 1) * 100 if mean_efficiency > 0 else 0,
            "% Diff Value": f"{(shock_efficiency / mean_efficiency - 1) * 100:.1f}%" if mean_efficiency > 0 else "N/A"
        },
        {
            "Metric": "Growth Rate",
            "Shock Country": shock_growth_rate,
            "Shock Value": f"{shock_growth_rate*100:.1f}%",
            "Mean": mean_growth,
            "Mean Value": f"{mean_growth*100:.1f}%",
            "Median": median_growth,
            "Median Value": f"{median_growth*100:.1f}%",
            "% Diff from Mean": (shock_growth_rate / mean_growth - 1) * 100 if mean_growth != 0 else 0,
            "% Diff Value": f"{(shock_growth_rate / mean_growth - 1) * 100:.1f}%" if mean_growth != 0 else "N/A"
        },
        {
            "Metric": "Connections",
            "Shock Country": shock_connections,
            "Shock Value": f"{shock_connections}",
            "Mean": mean_connections,
            "Mean Value": f"{mean_connections:.1f}",
            "Median": median_connections,
            "Median Value": f"{median_connections:.1f}",
            "% Diff from Mean": (shock_connections / mean_connections - 1) * 100 if mean_connections > 0 else 0,
            "% Diff Value": f"{(shock_connections / mean_connections - 1) * 100:.1f}%" if mean_connections > 0 else "N/A"
        },
        {
            "Metric": "Avg Tariff",
            "Shock Country": shock_avg_tariff,
            "Shock Value": f"{shock_avg_tariff:.2f}",
            "Mean": mean_tariff,
            "Mean Value": f"{mean_tariff:.2f}",
            "Median": median_tariff,
            "Median Value": f"{median_tariff:.2f}",
            "% Diff from Mean": (shock_avg_tariff / mean_tariff - 1) * 100 if mean_tariff > 0 else 0,
            "% Diff Value": f"{(shock_avg_tariff / mean_tariff - 1) * 100:.1f}%" if mean_tariff > 0 else "N/A"
        }
    ]
    
    return pd.DataFrame(comparison_data)