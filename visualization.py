import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
from model import TradeNetwork

def plot_network(net: TradeNetwork):
    # Use a larger k value for better spacing and a larger figure size
    pos = nx.spring_layout(net.G, seed=42, k=1.0)  # Increased k from 0.5 to 1.0 for better node spacing
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
        gdp_val = net.countries[cid].gdp
        # Scale node sizes more appropriately
        size = (10 + 5 * (gdp_val / 1e6)) if (gdp_val is not None and gdp_val > 0) else 10
        size = min(size, 40)  # Cap size for very large GDP values
        node_size.append(size)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_color, showscale=True),
        text=[f"Country {cid} (GDP: {net.countries[cid].gdp:.1f})" for cid in net.countries],
        hoverinfo="text",
    )
    fig = go.Figure(edge_trace)
    fig.add_trace(node_trace)
    # Improve layout with more space
    fig.update_layout(
        showlegend=False, 
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,  # Larger height
        width=800,   # Larger width
        plot_bgcolor='rgba(240,240,240,0.8)'  # Light gray background
    )
    # Use same scale for x and y axes
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(constrain='domain')
    return fig

def plot_histogram(data, nbins=20, title="Histogram"):
    return px.histogram(x=data, nbins=nbins, title=title)

def plot_scatter(df, x, y, title="Scatterplot", trendline=None):
    # First check if statsmodels is needed and available
    if trendline == 'ols':
        try:
            # Try to import statsmodels on demand
            import statsmodels.api as sm
            return px.scatter(df, x=x, y=y, trendline=trendline, title=title)
        except ImportError:
            # Fall back to no trendline if statsmodels is not available
            import streamlit as st
            st.warning("Trendline 'ols' requires statsmodels package. Displaying scatter plot without trendline.")
            return px.scatter(df, x=x, y=y, title=title)
    else:
        # No statsmodels needed
        return px.scatter(df, x=x, y=y, trendline=trendline, title=title)

def plot_line(df, x, y, title="Line Chart", error_y=None, markers=False, **kwargs):
    # Handle case where y is a list of columns (for multiple lines)
    if isinstance(y, list):
        fig = px.line(df, x=x, y=y, title=title, **kwargs)
    else:
        # Create a better formatted axis label from y column name
        y_label = y.replace('_', ' ').title() if isinstance(y, str) else y
        
        # Create figure with proper axis formatting
        if error_y:
            fig = px.line(
                df, x=x, y=y, 
                title=title,
                error_y=error_y,
                markers=markers,
                labels={
                    x: x.replace('param_value', 'Parameter Value').replace('_', ' ').title(),
                    y: y_label
                },
                **kwargs
            )
        else:
            fig = px.line(
                df, x=x, y=y, 
                title=title,
                markers=markers,
                labels={
                    x: x.replace('param_value', 'Parameter Value').replace('_', ' ').title(),
                    y: y_label
                },
                **kwargs
            )
            
    # Improve layout
    fig.update_layout(
        xaxis_title=x.replace('param_value', 'Parameter Value').replace('_', ' ').title(),
        legend_title_text="Data Series"
    )
    return fig

def plot_friendship_stats(net: TradeNetwork):
    """
    Plot friendship statistics for the country that applied the tariff shock,
    showing tariff changes before and after the shock.
    """
    # Check if friendship stats history exists
    if not hasattr(net, 'friendship_stats_history') or not net.friendship_stats_history:
        return go.Figure().update_layout(
            title="No tariff shock data available - run simulation with valid policy_shock parameter",
            height=400
        )
    
    # Get the history data
    history = net.friendship_stats_history
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Time Step': history['time'],
        'Outgoing Avg Tariff': history['outgoing_avg_tariff'],
        'Incoming Avg Tariff': history['incoming_avg_tariff'],
        'Countries in Same Bloc': history['same_bloc_count'],
        'Trading Partners': history['trading_partners']
    })
    
    # Create figure with multiple subplots - one for tariffs, one for bloc/partner counts
    fig = go.Figure()
    
    # Add outgoing tariff line
    fig.add_trace(go.Scatter(
        x=df['Time Step'],
        y=df['Outgoing Avg Tariff'],
        mode='lines+markers',
        name='Outgoing Avg Tariff',
        line=dict(color='red')
    ))
    
    # Add incoming tariff line
    fig.add_trace(go.Scatter(
        x=df['Time Step'],
        y=df['Incoming Avg Tariff'],
        mode='lines+markers',
        name='Incoming Avg Tariff',
        line=dict(color='blue')
    ))
    
    # Add vertical line at shock point
    if history['shock_point'] is not None:
        fig.add_vline(
            x=history['shock_point'], 
            line_width=2, 
            line_dash="dash", 
            line_color="green",
            annotation_text="Tariff Shock Applied",
            annotation_position="top right"
        )
        
        # Add annotation to explain the shock
        fig.add_annotation(
            x=history['shock_point'],
            y=max(max(df['Outgoing Avg Tariff']), max(df['Incoming Avg Tariff'])),
            text="Tariff Shock",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Create secondary axis for bloc and trading partner counts
    fig.add_trace(go.Scatter(
        x=df['Time Step'],
        y=df['Countries in Same Bloc'],
        mode='lines+markers',
        name='Countries in Same Bloc',
        line=dict(color='purple'),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Time Step'],
        y=df['Trading Partners'],
        mode='lines+markers',
        name='Trading Partners',
        line=dict(color='orange', dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title="Tariff and Relationship Changes Over Time",
        xaxis=dict(title="Time Step"),
        yaxis=dict(
            title="Average Tariff Rate",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            range=[0, 1]
        ),
        yaxis2=dict(
            title="Count",
            titlefont=dict(color="purple"),
            tickfont=dict(color="purple"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        height=600,
        width=800,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.7)'
        ),
        hovermode="x unified"
    )
    
    return fig


def plot_tariff_reciprocity(net: TradeNetwork, shock_id=None):
    """
    Plot a comparison of tariffs before and after reciprocal responses.
    Shows how other countries responded to the tariff shock.
    """
    if shock_id is None:
        # Try to get the shock_id from the friendship_stats_history
        if hasattr(net, 'friendship_stats_history') and net.friendship_stats_history:
            # Look for shock_id - first key in history
            for key in net.friendship_stats_history.keys():
                if key != 'time' and key != 'shock_point':
                    shock_id = int(key)
                    break
    
    if shock_id is None or shock_id not in net.countries:
        return go.Figure().update_layout(
            title="No valid shock country ID found",
            height=400
        )
    
    # Get all outgoing edges from shock country
    edges = list(net.G.out_edges(shock_id, data=True))
    
    # Prepare data for visualization
    target_countries = []
    outgoing_tariffs = []
    incoming_tariffs = []
    reciprocal_flags = []
    
    for _, target, data in edges:
        target_countries.append(f"Country {target}")
        outgoing_tariffs.append(data['tariff'])
        
        # Get incoming tariff if it exists
        if net.G.has_edge(target, shock_id):
            incoming_edge = net.G[target][shock_id]
            incoming_tariffs.append(incoming_edge['tariff'])
            reciprocal_flags.append(incoming_edge.get('reciprocal', False))
        else:
            incoming_tariffs.append(0)
            reciprocal_flags.append(False)
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for outgoing tariffs (from shock country to others)
    fig.add_trace(go.Bar(
        x=target_countries,
        y=outgoing_tariffs,
        name='Tariffs Applied by Shock Country',
        marker_color='red'
    ))
    
    # Add bar for incoming tariffs (from others to shock country)
    fig.add_trace(go.Bar(
        x=target_countries,
        y=incoming_tariffs,
        name='Reciprocal Tariffs',
        marker_color='blue'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Tariff Reciprocity - Country {shock_id}",
        xaxis=dict(title="Target Countries"),
        yaxis=dict(title="Tariff Rate", range=[0, 1]),
        barmode='group',
        height=500,
        width=800,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.7)'
        )
    )
    
    return fig
