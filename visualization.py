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

def plot_line(df, x, y, title="Line Chart", **kwargs):
    # Handle case where y is a list of columns (for multiple lines)
    if isinstance(y, list):
        return px.line(df, x=x, y=y, title=title, **kwargs)
    else:
        return px.line(df, x=x, y=y, title=title, **kwargs)
