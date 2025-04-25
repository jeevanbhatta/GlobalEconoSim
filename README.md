# GlobalEconoSim - Global Trade Network Simulation

This project simulates international trade networks and their evolution over time, accounting for geographic distance, political relationships (friendship), tariffs, and transaction costs.

## Key Features

- **Geographic Distance**: Countries located closer to each other have lower transaction costs
- **Friendship Dynamics**: Political relationships affect tariffs and evolve as trade happens
- **Network Statistics**: Comprehensive analysis of network structure including:
  - Clustering coefficients
  - Centrality measures (degree, betweenness)
  - Network diameter and path lengths
  - Community detection
  
## Interactive Simulation

The project includes a full-featured interactive Streamlit application that allows you to:

- Configure simulation parameters (countries, political blocs, tariffs, etc.)
- Run simulations and visualize trade networks in real-time
- Apply policy shocks and observe how they propagate through the network
- Compare network simulation with Mean-Field Approximation (MFA)
- Analyze economic metrics like GDP, poverty rates, and inequality

## Running the Interactive App

1. Install the required packages:
   ```bash
   pip install streamlit networkx numpy pandas plotly~=5.19 scipy
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run trade_app.py
   ```

3. The app will open in your web browser at http://localhost:8501

## Simulation Parameters

- **Countries**: Control the number of countries in the simulation (4-80)
- **Political blocs**: Group countries into alliance blocs with preferential trading
- **Tariff gap**: Set how much higher inter-bloc tariffs are compared to intra-bloc tariffs
- **Two-good world**: Enable comparative advantage with countries producing two goods with different efficiency
- **Policy shock**: Apply tariff changes to specific countries and observe ripple effects

## Components

- **Trade Network**: Directed graph with countries as nodes and trade relationships as edges
- **Friendship Matrix**: Tracks political relationships between countries
- **Transaction Costs**: Partially based on geographic distance between countries
- **Tariffs**: Based on political relationships (friendship matrix)

## Simulation Metrics

The simulation tracks and visualizes:

1. **Economic Indicators**:
   - Average tariffs
   - Transaction costs 
   - Friendship levels
   - GDP growth
   - Poverty rates
   - Inequality (Gini coefficient)

2. **Network Structure**:
   - Number of trade relationships
   - Network diameter
   - Clustering coefficient
   - Community formation

3. **Trade Hubs**:
   - Countries with high betweenness centrality
   - Major players in the global trade network

## File Structure

- `trade_app.py` - Main Streamlit application
- `model.py` - Core simulation primitives and engine
- `visualization.py` - Plotting and visualization functions
- `analytics.py` - Statistical analysis and advanced metrics
- `trade_stats.py` - Functions for analyzing trade network statistics

## Requirements

- Python 3.x
- NetworkX
- NumPy
- Matplotlib
- Pandas
- Plotly
- Streamlit
- Scipy
