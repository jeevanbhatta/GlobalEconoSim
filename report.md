
# Global Economic Simulation: Technical Report

## Overview

This report provides a detailed explanation of the Global Trade Simulation model, its technical implementation, key assumptions, and how to interpret its results. The simulation models international trade between countries as a network where trade flows are affected by tariffs, political blocs, and comparative advantages in production.

See the deployed version [here](https://tradesimulation.streamlit.app/). I'm hosting for free so the app gets inactive if not used for a while, but if you click the button, it makes the app active again in a few seconds. 

The source code can also be found in the [Github Repo](https://github.com/lifee77/GlobalEconoSim).

## 1. Technical Architecture

The simulation consists of four main components:

1. **Core Data Structures** (`model.py`)
   - Country representation
   - Trade network
   - Simulation engine

2. **Analytics** (`analytics.py`)
   - Mean-Field Approximation (MFA)
   - Parameter sensitivity analysis
   - Metrics calculation

3. **Visualization** (`visualization.py`)
   - Network graphs
   - Time series
   - Comparative charts

4. **UI/Interactive Layer** (`trade_app.py`)
   - Parameter controls
   - Visualization rendering
   - Results presentation

## 2. Model Primitives

### 2.1 Country

Each country is modeled as an entity with the following properties:

```python
@dataclass
class Country:
    cid: int                  # Country ID
    bloc: int                 # Political bloc affiliation
    gdp: float                # Gross Domestic Product
    population: int           # Population
    poverty_rate: float       # Percentage of population below poverty line
    dev_index: float          # Development index (0-1 scale)
    eff: Dict[str, float]     # Efficiency/productivity for goods A and B
    exports: float = 0.0      # Total exports
    imports: float = 0.0      # Total imports
    history: Dict[str, List[float]] = field(default_factory=lambda: {...})  # Time series data
```

Countries track their own economic indicators over time, allowing for time-series analysis of their development trajectory.

### 2.2 Trade Network

The trade network is implemented as a directed graph where:
- Nodes represent countries
- Edges represent trade relationships
- Edge weights include tariffs and distances

```python
class TradeNetwork:
    def __init__(self, countries, conn_intra, conn_inter, tariff_intra_mu, tariff_inter_mu, tariff_sd):
        self.countries = {c.cid: c for c in countries}
        self.G = nx.DiGraph()
        # ... network initialization ...
```

Key network initialization parameters:
- `conn_intra`: Probability of forming trade links within the same bloc
- `conn_inter`: Probability of forming trade links between different blocs
- `tariff_intra_mu`: Mean tariff rate for intra-bloc trade
- `tariff_inter_mu`: Mean tariff rate for inter-bloc trade
- `tariff_sd`: Standard deviation of tariff rates (randomness)

## 3. Simulation Engine

### 3.1 Initialization

The simulation is initialized by:
1. Creating `n` countries with random initial attributes
2. Assigning countries to political blocs (or no bloc at all)
3. Establishing trade links based on connection probabilities
4. Setting initial tariff rates based on bloc membership

### 3.2 Trade Flow Computation

For each time step, trade flows are computed using a modified gravity model:

$$\text{Flow}_{i,j,g} = \frac{\text{Productivity}_{i,g} \times \text{GDP}_i \times \text{GDP}_j}{\text{Distance}_{i,j}^2} \times (1 - \text{Tariff}_{i,j})^{\text{PriceElasticity}}$$

Where:
- $i,j$ are country indices
- $g$ is a good (A or B)
- Flow is capped to reasonable bounds to prevent numerical instability

```python
def compute_trade_flows(self, goods: List[str], price_elast: float = 1.0):
    for u, v, data in self.G.edges(data=True):
        c_u, c_v = self.countries[u], self.countries[v]
        tariff, dist = data["tariff"], data["dist"]
        total_flow = 0.0
        for g in goods:
            prod = c_u.eff[g]
            # Scale down GDP values to prevent overflow
            gdp_u_scaled = c_u.gdp / 1e6
            gdp_v_scaled = c_v.gdp / 1e6
            flow = np.clip((prod * gdp_u_scaled * gdp_v_scaled) / (dist ** 2), 0, 1e12)
            flow *= (1 - tariff) ** price_elast
            # ... bounds checking ...
            total_flow += flow
        data["flow"] = total_flow
```

This gravity model is consistent with empirical observations that trade volume:
- Increases with the economic size of trading partners
- Decreases with distance
- Is affected by trade barriers (tariffs)

### 3.3 Economic Growth Calculation

Each country's GDP growth is computed based on:
1. A base growth rate
2. Trade balance effects

```python
# Calculate trade balance with safeguards
trade_balance = 0.3 * exports - 0.2 * imports

# Limit GDP growth to prevent exponential explosion
# Use tanh to limit growth between -0.3 and +0.3
raw_growth = trade_balance / max(c.gdp, 1e3)
bounded_growth = 0.02 + 0.3 * np.tanh(raw_growth)  # Base growth + bounded trade effect

# Apply GDP growth with safety checks
c.gdp = min(c.gdp * (1 + bounded_growth), 1e15)  # Cap GDP at reasonable maximum
```

# Key assumptions:
- Base growth rate of 2% (0.02). This is considered normal growth too. [(Corporate Finance Institute, n.d.)](https://corporatefinanceinstitute.com/resources/economics/economic-growth-rate/#:~:text=For%20a%20developed%20economy%2C%20an,which%20leads%20to%20increased%20spending.)
- Exports contribute positively to growth (multiplier: 0.3)
- Imports have a smaller negative effect (multiplier: 0.2)
- Growth is bounded by a hyperbolic tangent function to ensure stability
- Hard caps prevent unrealistic values
- My initial assumption for **one step** in the simulation was a day, but that's not realistic as diplomatic processes take much longer. Realistically each step should be **1 week**.

# All other Assumptions

Where there are many economic theories that this simulation is based off of. These are a few of them that are applied. See notes for where these assumptions are simplified/redundant.

## A. Comparative Advantage 

- **Ricardo's Principle**: Each country has a comparative advantage in at least one good, as per Ricardian trade theory. This principle does not apply in the simulation as we have a two goods world with more than 2 countries.
- **Productivity Heterogeneity**: Countries draw random productivity values for goods A and B, creating natural comparative advantages.

## B. Trade Regimes and Tariffs

- **Tariff Continuum**: Countries exist on a spectrum from autarky (no trade) to free trade, represented by tariff values from 0 (free) to 1 (prohibitive). In real life we barely have any country with either of them. North Korea might be an extreme example but they still do have trade with China and Russia (and some assistance from South Korea).
- **Political Blocs**: The simulation models preferential trade agreements through lower intra-bloc tariffs. This aggregates two things that occur in real life
    - Trade agreements like [North American Trade Agreement](https://ustr.gov/trade-agreements/free-trade-agreements/united-states-mexico-canada-agreement) or the [EU zone] (https://policy.trade.ec.europa.eu/eu-trade-relationships-country-and-region/negotiations-and-agreements_en).
    - Friendships & Neighboring Status: Neighboring countries and allies even without special blocs are more likely to trade with each other.
- **Reciprocal Tariffs**: Higher tariffs from country A to B increase the probability of retaliatory tariffs from B to A, as seen in real-world trade wars. A major thing that I **ignored** here is the rade flow between the countries. Ideally if country B buys 90% of country A's goods, they would not like to go on a trade war. However, each othe treat each other the same in terms of recrprocal tarrif. And the reciprocal tarrifs are lower than original as the initial tarrif has already done economic damage to both countries.
- **Tariff Incidence**: The price elasticity parameter determines how tariffs affect trade flows, reflecting burden-sharing between producers and consumers.

## C. Transaction Costs

- **Distance Effect**: Trade flows follow a gravity model pattern: proportional to GDP and inversely proportional to squared distance.
- **Transport Modes**: While not explicitly modeled, the distance parameter implicitly captures the higher costs of trade for distant countries.
- **Landlocked Disadvantage**: Not directly modeled but implied by the distance parameter (could be extended to add specific landlocked penalties).

## D. Network Structure

- **Initial Connectivity**: Trade links form probabilistically, with higher chances within political blocs.
- **Trade Flow Evolution**: Links strengthen or weaken over time based on tariffs and economic growth.
- **Network Effects**: The model captures cascade effects where policy changes in one country affect the entire network.

## E. Economic Growth Dynamics

- **Trade-Driven Growth**: GDP growth is partially determined by trade balance, reflecting export-led growth theories.
- **Poverty Reduction**: Growth reduces poverty rates, following empirical evidence that trade openness tends to reduce poverty. This is based on the fact that employment increases and the cost of goods and services decreases. But, it's not guranteed that poverty decreases, specially when automation kicks in.
- **Increasing Returns**: While not explicitly modeled, the positive feedback between trade and growth captures elements of Krugman's increasing returns theory.

## F. Mean-Field Approximation (MFA)

- **Uniform Interaction**: The MFA replaces specific bilateral relationships with average interactions.
- **Non-Linear Divergence**: We assume that network effects create structural divergences from mean-field predictions.
- **Sigmoid Function**: The MFA model uses an S-shaped curve to relate aggregate variables, reflecting common economic transitions.

## G. Simplifications and Limitations

- **No Exchange Rates**: The model assumes a single global currency for simplicity.
- **No Resource Constraints**: The model ignores natural resource endowments that impact real-world trade patterns.
- **No Political Disruptions**: Wars, sanctions, and other political events are not entirely modeled. A country can have policy shocks (tarrifs or subsidy) but that applies to all other cuntries. 
- **No Dynamic Comparative Advantage**: We do not model the evolution of national capabilities over time.
- **No Capital Flows**: The model focuses on trade in goods.
- **Migrations** : The model also does not account for migration of both skilled laborers. 
- **Subsidy**: Tarrifs often come with subsidy to enhance local manufacturing or services. I have ignored that as we do not know the country's budget. But also, such processes take years and the simulation runs abrely for days.


## H. Welfare Implications

- **GDP Benefits**: The model assumes that, in general, more trade leads to higher GDP.
- **Distributional Effects**: The Gini coefficient tracks how trade patterns affect inequality **between** countries. Unfortunately,  We cannot account for vertical and horizontal inequality within a country in this Model.
- **Deadweight Loss**: Higher tariffs reduce total world GDP, reflecting deadweight losses from trade barriers.

### 3.4 Policy Shocks

The simulation allows for policy shocks, particularly changes in tariff rates:

```python
def apply_tariff_delta(self, cid: int, delta: float):
    # Apply tariff to all outgoing edges from the country
    for _, v, d in self.G.out_edges(cid, data=True):
        old_tariff = d["tariff"]
        d["tariff"] = np.clip(old_tariff + delta, 0, 1)
        # Mark edge as being under a reciprocal tariff regime
        d["reciprocal"] = True
        
        # ... update diplomatic relations ...
        
        # Add reciprocal tariffs from other countries
        # Other countries will respond with a partial reciprocal tariff (70% of original)
        if self.G.has_edge(v, cid):
            recip_edge = self.G[v][cid]
            old_recip_tariff = recip_edge["tariff"]
            recip_delta = delta * 0.7  # 70% reciprocal response
            recip_edge["tariff"] = np.clip(recip_edge["tariff"] + recip_delta, 0, 1)
            recip_edge["reciprocal"] = True
```

Key features:
- Tariff changes apply to all outgoing edges from a country
- Other countries respond with a 70% reciprocal tariff change
- All tariff values are bounded between 0 and 1
- Diplomatic relations are updated based on tariff changes

### 3.5 Political Bloc Formation

The simulation models the dynamic formation of political blocs based on trade relations:

```python
def allow_bloc_formation(self, tariff_threshold=0.4):
    """
    Form new blocs in response to high tariffs.
    Countries with low mutual tariffs might form a new bloc.
    """
    # Find pairs of countries with low mutual tariffs
    # ...
    # If both tariffs are below threshold, consider forming a bloc
    if i_to_j_tariff < tariff_threshold and j_to_i_tariff < tariff_threshold:
        avg_tariff = (i_to_j_tariff + j_to_i_tariff) / 2
        potential_bloc_pairs.append((i, j, avg_tariff))
    # ...
    # Apply new bloc assignments
```

Countries with mutually low tariffs may form new blocs, modeling the real-world tendency of countries to form trade agreements and economic unions when trade barriers are low.

## 4. Mean-Field Approximation (MFA)

### 4.1 MFA Theory

The Mean-Field Approximation (MFA) is a technique from statistical physics that simplifies complex interaction systems by replacing all pairwise interactions with an average "field." In our trade simulation:

- The full network simulation tracks individual trade links between specific countries
- The MFA treats each country as trading with an "average world economy"

This allows us to evaluate the importance of network structure by comparing the fully heterogeneous model with its homogeneous approximation.

### 4.2 MFA Implementation

```python
def mfa_series(network, steps):
    """
    Generate a Mean Field Approximation time series for comparison with full network sim.
    MFA ignores network structure and treats every country as interacting with the 'average' country.
    """
    # Get initial values
    countries = list(network.countries.values())
    
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
```

Key MFA assumptions:
- Base growth rate of 2%
- Trade adds up to 3% additional growth
- A cyclical component simulates business cycles
- No heterogeneity in tariffs or political blocs
- No network effects

### 4.3 MFA vs. Full Simulation

The MFA differs from the full simulation in several key ways:

| Feature | Full Simulation | MFA |
|---------|----------------|-----|
| Trade connections | Heterogeneous, based on blocs | Homogeneous, all countries connected equally |
| Tariffs | Vary by country pair and bloc membership | Implicit in average growth rate |
| Shocks | Propagate through network with ripple effects | Affect overall growth directly |
| Comparative advantage | Modeled explicitly through productivity differences | Not modeled |
| Inequality | Emerges from network structure | Not captured |

## 5. Parameter Sensitivity Analysis

The simulation includes tools for parameter sensitivity analysis, which helps identify:
- Critical parameter thresholds
- Regions of stability/instability
- Relative importance of different model components

```python
def parameter_sensitivity_analysis(param_name, param_range, fixed_params, target_metric, steps, num_replicates=1):
    """
    Run sensitivity analysis on a parameter by varying it while keeping others fixed.
    Returns a DataFrame with results.
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
            # ... run simulation ...
            # ... calculate metrics ...
            replicate_results['final_values'].append(final_value)
            replicate_results['growth_rates'].append(growth_rate)
        
        # Calculate mean and standard deviation across replicates
        final_value_mean = np.mean(replicate_results['final_values'])
        final_value_std = np.std(replicate_results['final_values'])
        
        # ... store results ...
    
    return pd.DataFrame(results)
```

This allows for rigorous testing of how the model responds to different parameters.


## 6. Interpreting Simulation Results
I'm running the simulation with the following default values:
Countries: 80, Political Blocs: 3, Time Steps: 300 weeks, Intra Bloc link (trade) probability: 0.8, Inter bloc link probability: 0.5, Tarrif gap (between intra and inter): 0.3, Tarrif S.D.: 0.15. I have set tarrif shock (by a special country to 0). ANd this is a two good world.
This results in the follwoing higher order effects.
![World GDP with initial Parameters](resources/GDP_initial_aram.png)
_**Figure 1**World GDP (Y-axis) along with simulation runs (X axis- weeks). Each country has initial GDP, which increases over time as trade continues. GDP = C + I + G + (X - M), where C,I,G,X, and M represent Consuemr spending, investment, government spending, export and import respectively. This is an interactive graph. Please use the [website](https://tradesimulation.streamlit.app/) to see it best._

This result is expected (even in real life) with trade growth as consumers get cheaper goods (countries product the most efficient good). The tarrifs (minimal so far) also increase government budget. Consumer spending and increasing taxes also contribute to higher budget, eventually increasing governemnt spending- which generates employment and increases consumer spending again making a reinforcing cycle. 

### Tarrif Distribution:
The tarrif distribution here is an **emergent property**. There are multiple factors playing a role.

- Distribution for in-bloc countries.
- Distribution for countriesoutside the bloc.
- Reciprocal tarrifs.
The initial mean is 0.1 as countries do have some sort of tarrif to protect vulnerable local industries (In real life tarrifs generally are sectoral, we are just averaging those out here). Before reciprocal tarrifs kick in, this is how the calculation is done:

- tariff_gap = 0.3 → gap/2 = 0.15
- tariff_sd = 0.15

We get;

- intra‑bloc mean = 0.10 − 0.15 = −0.05 → clipped to 0.0
- inter‑bloc mean = 0.10 + 0.15 = 0.25

and each draw is from N(0.0, 0.15) or N(0.25, 0.15), then clipped to [0,1].

The combination of those two Normal distribution and reciprocal tarrif gives a dstribution like this:

![Tarrif Distribution](resources/tarrif_distribution_init.png)

_**Figure 2.** Tarrif distribution in the Simulated Global Economy. The X-axis denotes tarrif value (with 1 being trade-restrictive and 0 being free trade) and the Y axis being trade relations (all the edges in the system combined)._

If we draw these two normal disribution, we would get these graphs;
![Normal Distributions with Cutoff at 0](resources/normal_dist.png)
_**Figure 3.** Histogram for Normal Distributions N(0.0, 0.15) in the left and N(0.25, 0.15) in the right. Both histograms are clipped at 0 tarrif assuming countries would not subsidize foreign goods (there are some exceptions in real life; eg: Nepal subsidizes farm equipments). 80 countries are used for both histograms._

### Network Structure


The trade network visualization shows:
- Node color: Political bloc membership
- Node size: GDP (larger = higher GDP)
- Edge width: Trade flow volume
- Edge style: Solid for normal trade, dashed for reciprocal tariff relationships



**Interpretation:** Look for clustering of nodes by color (political blocs), the formation of trade communities, and the distribution of node sizes. Highly connected nodes usually become wealthier over time.

### 6.2 Economic Indicators

#### 6.2.1 World GDP

The simulation tracks total world GDP over time, comparing the full simulation with the MFA.

**Interpretation:** Divergence between simulation and MFA indicates the importance of network effects. When the lines diverge significantly, it suggests that heterogeneity and network structure have major impacts on economic outcomes.

#### 6.2.2 Poverty Rate

The poverty rate is modeled as being reduced by economic growth:

```python
growth_effect = np.clip(bounded_growth - 0.01, -0.5, 0.5)  # Limit effect
c.poverty_rate = max(c.poverty_rate * (1 - 0.3 * growth_effect), 0.01)
```

**Interpretation:** Track how poverty rates respond to policy changes. Countries with higher growth typically see faster poverty reduction, but network effects can lead to uneven results.

#### 6.2.3 Gini Coefficient

The simulation calculates the Gini coefficient to measure income inequality:

```python
def gini(x):
    n = len(x)
    if n <= 1:
        return 0
    s = sum(i * x[i] for i in range(n))
    return (2 * s / (n * sum(x)) - (n + 1) / n)
```

**Interpretation:** Rising Gini coefficients indicate increasing inequality. Compare the simulation Gini with the MFA Gini to see how network structure affects inequality.

### 6.3 Policy Shock Effects

When applying policy shocks (e.g., tariff changes):

**Interpretation:**
- Look for ripple effects through the network
- Compare the shocked country's growth rate with its neighbors
- Observe how the shock affects network fragmentation
- Evaluate changes in global indicators (GDP, poverty, Gini)

### 6.4 Fragmentation Analysis

The simulation measures network fragmentation by removing weak trade links:

```python
# fragmentation: components after pruning weak edges
flows_array = np.array(flows)
threshold = np.percentile(flows_array, 40)  # weakest 40% removed
weak_edges = [(u, v) for u, v, d in G_copy.edges(data=True) if d.get("flow", 0) < threshold]
G_copy.remove_edges_from(weak_edges)
comps = nx.number_weakly_connected_components(G_copy)
```

**Interpretation:** More components indicate a fragmented world economy. High fragmentation often occurs when tariff gaps between blocs are high, representing a world divided into distinct trading communities.

## 7. Technical Implementation Details

### 7.1 Numerical Stability

Several measures ensure numerical stability:
- GDP values are scaled down during flow calculations: `gdp_u_scaled = c_u.gdp / 1e6`
- Growth rates are bounded using hyperbolic tangent: `bounded_growth = 0.02 + 0.3 * np.tanh(raw_growth)`
- Hard caps on values: `c.gdp = min(c.gdp * (1 + bounded_growth), 1e15)`
- Minimum values for poverty rates: `c.poverty_rate = max(c.poverty_rate * (1 - 0.3 * growth_effect), 0.01)`

### 7.2 Randomness

Random elements include:
- Initial country GDP, population, poverty rates
- Initial productivity for goods A and B
- Trade link formation (probabilistic)
- Initial tariff rates (drawn from normal distribution)

### 7.3 Performance Considerations

For larger simulations:
- Edge count grows as O(n²) with country count
- Computation time scales approximately as O(n² × steps)
- Memory usage is dominated by the history data structures

## 8. Extensions and Limitations

### 8.1 Possible Extensions

The simulation could be extended to include:
- More realistic trade models (e.g., Heckscher-Ohlin, New Trade Theory)
- Financial markets and capital flows
- Exchange rate dynamics
- Supply chain modeling
- Climate change impacts
- Demographic changes
- Technology diffusion
- More sophisticated political dynamics

### 8.2 Current Limitations

The current model has several limitations:
- Simplified growth model without production factors
- Limited historical calibration to real-world data
- No geographic positioning of countries
- Limited number of goods (maximum 2)
- Stylized rather than realistic political dynamics
- No inclusion of non-economic factors (culture, institutions)

## 9. Conclusion

The Global Trade Simulation provides a flexible platform for exploring the dynamics of international trade, policy shocks, and economic development. Its key strengths are:

1. **Network-based approach** that captures the complex interdependencies of the global economy
2. **Mean-Field Approximation comparison** that highlights the importance of network structure
3. **Parameter sensitivity analysis** that helps identify critical thresholds and relationships
4. **Interactive visualization** that makes complex economic dynamics interpretable
5. **Modular design** that allows for extensions and modifications

While simplified compared to real-world trade dynamics, the simulation captures many key features of international economic relations and provides intuitive insights into how trade networks, tariff policies, and political blocs shape global economic outcomes.
