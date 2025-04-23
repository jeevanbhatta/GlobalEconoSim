import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

def plot_network(network):
    """
    Plot the trade network using Plotly
    """
    G = network.G
    
    # Create positions using Kamada-Kawai layout for better spread
    pos = nx.kamada_kawai_layout(G)
    
    # Create node traces
    bloc_colors = px.colors.qualitative.Bold
    node_traces = []
    
    for node, data in G.nodes(data=True):
        c = network.countries[node]
        bloc = c.bloc
        color = bloc_colors[bloc % len(bloc_colors)]
        
        # Size node by GDP
        size = np.sqrt(c.gdp) / 25
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0]],
            y=[pos[node][1]],
            text=[f"Country {node}<br>Bloc: {bloc}<br>GDP: {c.gdp:.0f}<br>Pop: {c.population:,}<br>Poverty: {c.poverty_rate:.1%}"],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=1, color='black')
            ),
            name=f"Country {node}",
            showlegend=False,
            hoverinfo='text'
        )
        node_traces.append(node_trace)
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        u, v, d = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Use flow for edge width, but apply a minimum
        flow = d.get('flow', 0)
        width = 1 + 5 * np.log1p(flow) / 10 if flow > 0 else 0.5
        
        # Curved edges (add midpoint offset)
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2
        offset = 0.05  # curve strength
        
        # For reciprocal edges, use a different color
        dash = 'dash' if d.get('reciprocal', False) else 'solid'
        color = 'purple' if d.get('reciprocal', False) else 'rgba(180,180,180,0.7)'
        
        # Create curved line
        edge_trace = go.Scatter(
            x=[x0, xm, x1],
            y=[y0, ym + offset, y1],
            mode='lines',
            line=dict(width=width, color=color, dash=dash),
            hoverinfo='text',
            text=f"Flow: {flow:.2f}<br>Tariff: {d.get('tariff', 0):.2f}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Add legend for bloc colors
    blocs_seen = set()
    for node in G.nodes():
        bloc = network.countries[node].bloc
        if bloc not in blocs_seen:
            blocs_seen.add(bloc)
            color = bloc_colors[bloc % len(bloc_colors)]
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"Bloc {bloc}",
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Trade Network", 
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            title="Political Blocs",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        plot_bgcolor='white'
    )
    
    return fig

def plot_histogram(data, nbins=20, title=""):
    """
    Plot a histogram using Plotly Express
    """
    fig = px.histogram(data, nbins=nbins, title=title)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Value",
        yaxis_title="Count",
        bargap=0.1
    )
    return fig

def plot_scatter(df, x, y, trendline=None, title=""):
    """
    Plot a scatter plot with optional trendline using Plotly Express
    """
    fig = px.scatter(df, x=x, y=y, trendline=trendline, title=title)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x,
        yaxis_title=y
    )
    return fig

def plot_line(data, x, y, error_y=None, markers=False, title=""):
    """
    Plot a line chart with optional error bars
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    if isinstance(y, list):
        # Multiple y columns
        fig = px.line(data, x=x, y=y, title=title)
        if markers:
            for trace in fig.data:
                trace.mode = 'lines+markers'
    else:
        # Single y column with possible error bars
        if error_y:
            fig = px.line(data, x=x, y=y, error_y=error_y, title=title)
            # Add markers if requested
            if markers:
                fig.update_traces(mode='lines+markers')
        else:
            fig = px.line(data, x=x, y=y, title=title)
            if markers:
                fig.update_traces(mode='lines+markers')
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x,
        yaxis_title=y if isinstance(y, str) else "Value"
    )
    
    return fig

def plot_bar(data, x, y, color=None, barmode='group', title=""):
    """
    Plot a bar chart
    """
    fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=title)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color if color else None
    )
    return fig

def plot_heatmap(data, x, y, z, title=""):
    """
    Plot a heatmap for correlations or other 2D data
    """
    fig = px.imshow(data, x=x, y=y, color_continuous_scale="RdBu_r", title=title)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x,
        yaxis_title=y
    )
    return fig
