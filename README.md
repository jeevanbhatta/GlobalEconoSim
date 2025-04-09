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

2. **Network Structure**:
   - Number of trade relationships
   - Network diameter
   - Clustering coefficient
   - Community formation

3. **Trade Hubs**:
   - Countries with high betweenness centrality
   - Major players in the global trade network

## Extended Analysis

The notebook includes additional analysis tools:
- Distance vs. transaction cost correlation
- Community detection and visualization
- Trade hub identification
- Centrality analysis for identifying key countries in the network

## Requirements

- Python 3.x
- NetworkX
- NumPy
- Matplotlib
- Pandas
- Seaborn
- scipy
- python-louvain (optional, for community detection algorithm)

## Getting Started

1. Install required packages:
   ```
   pip install networkx numpy matplotlib pandas seaborn scipy
   pip install python-louvain  # Optional, for community detection
   ```

2. Run the Jupyter notebook:
   ```
   jupyter notebook trade_n_countries.ipynb
   ```

3. Execute all cells to run the simulation and view the results
