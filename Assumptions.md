# Economic Assumptions in the Global Trade Simulation

This document outlines the theoretical assumptions and principles from international trade economics that underpin our simulation model.

## 1. Comparative Advantage and Specialization

- **Ricardo's Principle**: Each country has a comparative advantage in at least one good, as per Ricardian trade theory.
- **Productivity Heterogeneity**: Countries draw random productivity values for goods A and B, creating natural comparative advantages.
- **Specialization Emergence**: The model allows for specialization patterns to emerge organically from these productivity differences.

## 2. Trade Regimes and Tariffs

- **Tariff Continuum**: Countries exist on a spectrum from autarky (no trade) to free trade, represented by tariff values from 0 (free) to 1 (prohibitive).
- **Political Blocs**: The simulation models preferential trade agreements through lower intra-bloc tariffs.
- **Reciprocal Tariffs**: Higher tariffs from country A to B increase the probability of retaliatory tariffs from B to A, as seen in real-world trade wars.
- **Tariff Incidence**: The price elasticity parameter determines how tariffs affect trade flows, reflecting burden-sharing between producers and consumers.

## 3. Transaction Costs

- **Distance Effect**: Trade flows follow a gravity model pattern: proportional to GDP and inversely proportional to squared distance.
- **Transport Modes**: While not explicitly modeled, the distance parameter implicitly captures the higher costs of trade for distant countries.
- **Landlocked Disadvantage**: Not directly modeled but implied by the distance parameter (could be extended to add specific landlocked penalties).

## 4. Network Structure

- **Initial Connectivity**: Trade links form probabilistically, with higher chances within political blocs.
- **Trade Flow Evolution**: Links strengthen or weaken over time based on tariffs and economic growth.
- **Network Effects**: The model captures cascade effects where policy changes in one country affect the entire network.

## 5. Economic Growth Dynamics

- **Trade-Driven Growth**: GDP growth is partially determined by trade balance, reflecting export-led growth theories.
- **Poverty Reduction**: Growth reduces poverty rates, following empirical evidence that trade openness tends to reduce poverty.
- **Increasing Returns**: While not explicitly modeled, the positive feedback between trade and growth captures elements of Krugman's increasing returns theory.

## 6. Mean-Field Approximation (MFA)

- **Uniform Interaction**: The MFA replaces specific bilateral relationships with average interactions.
- **Non-Linear Divergence**: We assume that network effects create structural divergences from mean-field predictions.
- **Sigmoid Function**: The MFA model uses an S-shaped curve to relate aggregate variables, reflecting common economic transitions.

## 7. Simplifications and Limitations

- **No Exchange Rates**: The model assumes a single global currency for simplicity.
- **No Resource Constraints**: The model ignores natural resource endowments that impact real-world trade patterns.
- **No Political Disruptions**: Wars, sanctions, and other political events are not modeled.
- **No Dynamic Comparative Advantage**: We do not model the evolution of national capabilities over time.
- **No Capital Flows**: The model focuses on trade in goods, excluding capital account dynamics.

## 8. Statistical Analysis Principles

- **Multiple Replicates**: Parameter sensitivity analysis runs multiple simulations to capture stochastic variability.
- **Equilibrium Tests**: Network fragmentation and Gini coefficient stability are used as indicators of reaching equilibrium.
- **Model Selection**: When fitting MFA curves, we use AIC/BIC criteria to balance complexity and fit.
- **Error Analysis**: We track divergence between MFA and simulation to quantify network effects.

## 9. Welfare Implications

- **GDP Benefits**: The model assumes that, in general, more trade leads to higher GDP.
- **Distributional Effects**: The Gini coefficient tracks how trade patterns affect inequality between countries.
- **Deadweight Loss**: Higher tariffs reduce total world GDP, reflecting deadweight losses from trade barriers.