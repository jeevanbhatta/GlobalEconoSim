# %% [markdown]
# # International Trade Simulation: Tariffs, Substitutes, and Retaliation
#
# This simulation models trade between Country A and Country B.
# - Country B exports Electronics to Country A.
# - Country A exports Agricultural Goods to Country B.
#
# We compare two scenarios for Country A's policy on Electronics imports:
# 1. Free Trade: No tariffs.
# 2. Tariff: Country A imposes a tariff on Electronics. This triggers Country B to impose a retaliatory tariff on Agricultural Goods.
#
# Key features:
# - Randomness: Exchange rate fluctuations, demand shocks for both goods.
# - Domestic Substitute: Country A has a domestic substitute for Electronics, making import demand sensitive to price increases caused by tariffs.
# - Monte Carlo: The simulation is run many times to understand the distribution of outcomes.

# %% [markdown]
# ## Cell 1: Setup - Imports and Parameters

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- Simulation Parameters ---
NUM_RUNS = 1000  # Number of Monte Carlo runs
NUM_MONTHS = 60  # Simulation duration (e.g., 5 years)

# --- Economic Parameters ---
# Exchange Rate (Units of A's currency per 1 unit of B's currency)
BASE_EXCHANGE_RATE = 1.0
EXCHANGE_RATE_VOLATILITY = 0.03 # Std dev of monthly percentage change

# Electronics (B -> A)
BASE_PRICE_ELEC_B = 100       # Price in Country B's currency
BASE_DEMAND_ELEC_A = 1000     # Base monthly demand in A at reference price
DEMAND_ELASTICITY_ELEC = 1.5  # Price elasticity of demand for electronics
DEMAND_VOLATILITY_ELEC = 0.10 # Std dev of monthly demand shock (as fraction)
SUPPLY_VOLATILITY_ELEC = 0.05 # Std dev of monthly supply shock (as fraction)
BASE_SUPPLY_ELEC_B = 1100     # Base monthly supply from B
DOMESTIC_SUBSTITUTE_PRICE_A = 115 # Price of domestic substitute in A's currency
SUBSTITUTE_PENALTY_FACTOR = 0.4 # Demand multiplier if import price > substitute

# Agricultural Goods (A -> B)
BASE_PRICE_AGRI_A = 50        # Price in Country A's currency
BASE_DEMAND_AGRI_B = 2000     # Base monthly demand in B at reference price
DEMAND_ELASTICITY_AGRI = 0.8  # Price elasticity of demand for agri goods
DEMAND_VOLATILITY_AGRI = 0.15 # Std dev of monthly demand shock
BASE_SUPPLY_AGRI_A = 2200     # Base monthly supply from A (simplified: assumed sufficient often)

# --- Policy Parameters ---
TARIFF_RATE_A_ON_ELEC = 0.15  # 15% tariff by A on Electronics
RETALIATORY_TARIFF_RATE_B = 0.10 # 10% tariff by B on Agri Goods

# Reference prices used for elasticity calculations (can be same as base prices)
# These represent the price point at which base demand is defined.
BASE_PRICE_ELEC_A_REF = BASE_PRICE_ELEC_B * BASE_EXCHANGE_RATE
BASE_PRICE_AGRI_B_REF = BASE_PRICE_AGRI_A / BASE_EXCHANGE_RATE


# %% [markdown]
# ## Cell 2: Simulation Function Definition

# %%
def run_trade_simulation(num_months, scenario):
    """
    Runs a single simulation of trade for num_months.

    Args:
        num_months (int): The number of monthly steps to simulate.
        scenario (str): 'Free Trade' or 'Tariff'. Determines if tariffs are applied.

    Returns:
        dict: A dictionary containing total outcomes for the run.
              Keys: 'total_elec_volume', 'total_agri_volume',
                    'total_elec_tariff_revenue_A', 'total_agri_tariff_revenue_B',
                    'avg_elec_price_A', 'avg_agri_price_B'
    """

    # --- Initialize state variables for this run ---
    current_exchange_rate = BASE_EXCHANGE_RATE
    monthly_elec_volume = []
    monthly_agri_volume = []
    monthly_elec_tariff_revenue_A = []
    monthly_agri_tariff_revenue_B = []
    monthly_elec_price_A_post_tariff = [] # Store the price consumer faces in A
    monthly_agri_price_B_post_tariff = [] # Store the price consumer faces in B

    # --- Determine tariffs based on scenario ---
    tariff_A = 0.0
    tariff_B = 0.0
    if scenario == 'Tariff':
        tariff_A = TARIFF_RATE_A_ON_ELEC
        tariff_B = RETALIATORY_TARIFF_RATE_B # Retaliation occurs

    # --- Simulation Loop ---
    for month in range(num_months):
        # 1. Update Exchange Rate (Random Walk element)
        rate_change = np.random.normal(0, EXCHANGE_RATE_VOLATILITY)
        current_exchange_rate *= (1 + rate_change)
        # Add a floor/ceiling or mean reversion if desired, but simple random walk for now
        current_exchange_rate = max(0.1, current_exchange_rate) # Prevent negative/zero

        # 2. Electronics Trade (B -> A)
        # Supply from B
        supply_shock_elec = np.random.normal(1, SUPPLY_VOLATILITY_ELEC)
        supply_elec = BASE_SUPPLY_ELEC_B * supply_shock_elec
        supply_elec = max(0, supply_elec)

        # Price in A
        price_elec_in_A_pre_tariff = BASE_PRICE_ELEC_B * current_exchange_rate
        price_elec_in_A_post_tariff = price_elec_in_A_pre_tariff * (1 + tariff_A)
        monthly_elec_price_A_post_tariff.append(price_elec_in_A_post_tariff)

        # Demand in A (incorporating elasticity, substitute, and shock)
        demand_shock_elec = np.random.normal(1, DEMAND_VOLATILITY_ELEC)
        price_ratio_elec = price_elec_in_A_post_tariff / BASE_PRICE_ELEC_A_REF
        demand_elec = BASE_DEMAND_ELEC_A * (price_ratio_elec ** (-DEMAND_ELASTICITY_ELEC))

        # Apply substitute penalty if import price is higher
        if price_elec_in_A_post_tariff > DOMESTIC_SUBSTITUTE_PRICE_A:
             demand_elec *= SUBSTITUTE_PENALTY_FACTOR

        demand_elec *= demand_shock_elec # Apply random shock
        demand_elec = max(0, demand_elec) # Demand cannot be negative

        # Volume Traded
        volume_elec = min(demand_elec, supply_elec)
        monthly_elec_volume.append(volume_elec)

        # Tariff Revenue for A
        revenue_A = volume_elec * price_elec_in_A_pre_tariff * tariff_A
        monthly_elec_tariff_revenue_A.append(revenue_A)

        # 3. Agricultural Goods Trade (A -> B)
        # Supply from A (simplified - assume A can meet demand for now)
        # More complex: could link supply_agri to its own price or have shocks
        supply_agri = BASE_SUPPLY_AGRI_A # Or add random shock: * np.random.normal(1, SUPPLY_VOLATILITY_AGRI)
        supply_agri = max(0, supply_agri)

        # Price in B
        price_agri_in_B_pre_tariff = BASE_PRICE_AGRI_A / current_exchange_rate
        price_agri_in_B_post_tariff = price_agri_in_B_pre_tariff * (1 + tariff_B) # Retaliatory tariff
        monthly_agri_price_B_post_tariff.append(price_agri_in_B_post_tariff)

        # Demand in B (incorporating elasticity and shock)
        demand_shock_agri = np.random.normal(1, DEMAND_VOLATILITY_AGRI)
        price_ratio_agri = price_agri_in_B_post_tariff / BASE_PRICE_AGRI_B_REF
        demand_agri = BASE_DEMAND_AGRI_B * (price_ratio_agri ** (-DEMAND_ELASTICITY_AGRI))
        demand_agri *= demand_shock_agri # Apply random shock
        demand_agri = max(0, demand_agri)

        # Volume Traded
        volume_agri = min(demand_agri, supply_agri)
        monthly_agri_volume.append(volume_agri)

        # Tariff Revenue for B (due to retaliation)
        revenue_B = volume_agri * price_agri_in_B_pre_tariff * tariff_B
        monthly_agri_tariff_revenue_B.append(revenue_B)

    # --- Aggregate results for the run ---
    results = {
        'total_elec_volume': np.sum(monthly_elec_volume),
        'total_agri_volume': np.sum(monthly_agri_volume),
        'total_elec_tariff_revenue_A': np.sum(monthly_elec_tariff_revenue_A),
        'total_agri_tariff_revenue_B': np.sum(monthly_agri_tariff_revenue_B),
        'avg_elec_price_A': np.mean(monthly_elec_price_A_post_tariff) if monthly_elec_price_A_post_tariff else 0,
        'avg_agri_price_B': np.mean(monthly_agri_price_B_post_tariff) if monthly_agri_price_B_post_tariff else 0,
    }
    return results

# %% [markdown]
# ## Cell 3: Running the Monte Carlo Simulations

# %%
results_free_trade = []
results_tariff = []

print(f"Running {NUM_RUNS} simulations for each scenario...")

for i in range(NUM_RUNS):
    if (i + 1) % (NUM_RUNS // 10) == 0: # Print progress
        print(f"... completed {i+1}/{NUM_RUNS} runs")
    results_free_trade.append(run_trade_simulation(NUM_MONTHS, 'Free Trade'))
    results_tariff.append(run_trade_simulation(NUM_MONTHS, 'Tariff'))

print("Simulations complete.")

# Convert results lists to dictionaries of arrays for easier access
outcomes_free_trade = {key: np.array([res[key] for res in results_free_trade]) for key in results_free_trade[0]}
outcomes_tariff = {key: np.array([res[key] for res in results_tariff]) for key in results_tariff[0]}


# %% [markdown]
# ## Cell 4: Results Analysis - Statistics and Confidence Intervals

# %%
def calculate_stats(data_array):
    """Calculates mean, std dev, and 95% CI for the mean."""
    mean = np.mean(data_array)
    std_dev = np.std(data_array)
    ci = st.t.interval(0.95, len(data_array)-1, loc=mean, scale=st.sem(data_array))
    return mean, std_dev, ci

print("\n--- Simulation Results Analysis ---")

metrics = [
    ('total_elec_volume', 'Total Electronics Volume (B->A)'),
    ('total_agri_volume', 'Total Agri. Volume (A->B)'),
    ('total_elec_tariff_revenue_A', "Total Elec. Tariff Revenue (A)"),
    ('total_agri_tariff_revenue_B', "Total Agri. Tariff Revenue (B)"),
    ('avg_elec_price_A', "Avg. Electronics Price (in A)"),
    ('avg_agri_price_B', "Avg. Agri. Price (in B)"),
]

print("\n--- Free Trade Scenario ---")
stats_free_trade = {}
for key, name in metrics:
    if key in outcomes_free_trade: # Ensure key exists (revenues are 0 here)
        mean, std_dev, ci = calculate_stats(outcomes_free_trade[key])
        stats_free_trade[key] = {'mean': mean, 'std_dev': std_dev, 'ci': ci}
        print(f"{name}:")
        print(f"  Mean: {mean:,.2f}")
        print(f"  Std Dev: {std_dev:,.2f}")
        print(f"  95% CI: ({ci[0]:,.2f}, {ci[1]:,.2f})")

print("\n--- Tariff Scenario ---")
stats_tariff = {}
for key, name in metrics:
     if key in outcomes_tariff:
        mean, std_dev, ci = calculate_stats(outcomes_tariff[key])
        stats_tariff[key] = {'mean': mean, 'std_dev': std_dev, 'ci': ci}
        print(f"{name}:")
        print(f"  Mean: {mean:,.2f}")
        print(f"  Std Dev: {std_dev:,.2f}")
        print(f"  95% CI: ({ci[0]:,.2f}, {ci[1]:,.2f})")


# %% [markdown]
# ## Cell 5: Plotting Results - Histograms

# %%
def plot_histograms(key, name):
    """Plots histograms comparing Free Trade and Tariff scenarios for a given metric."""
    plt.figure(figsize=(10, 5))

    data_ft = outcomes_free_trade.get(key, np.array([0])) # Default to 0 if key missing (like revenue in FT)
    data_t = outcomes_tariff.get(key, np.array([0]))

    # Determine common sensible bins
    combined_data = np.concatenate((data_ft, data_t))
    if np.all(combined_data == 0): # Handle cases like revenue in FT
         bins=10
    else:
        bins = np.linspace(np.min(combined_data), np.max(combined_data), 50)


    plt.hist(data_ft, bins=bins, alpha=0.7, label='Free Trade', density=True)
    plt.hist(data_t, bins=bins, alpha=0.7, label='Tariff (+Retaliation)', density=True)

    mean_ft = stats_free_trade.get(key, {}).get('mean', 0)
    mean_t = stats_tariff.get(key, {}).get('mean', 0)
    plt.axvline(mean_ft, color='blue', linestyle='dashed', linewidth=1, label=f'Mean FT: {mean_ft:,.0f}')
    plt.axvline(mean_t, color='red', linestyle='dashed', linewidth=1, label=f'Mean Tariff: {mean_t:,.0f}')

    plt.title(f'Distribution of {name} ({NUM_RUNS} runs)')
    plt.xlabel(name)
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

print("\n--- Histograms ---")
# Plot for key metrics
plot_histograms('total_elec_volume', 'Total Electronics Volume (B->A)')
plot_histograms('total_agri_volume', 'Total Agri. Volume (A->B)')
plot_histograms('total_elec_tariff_revenue_A', 'Total Elec. Tariff Revenue (A)')
# plot_histograms('total_agri_tariff_revenue_B', 'Total Agri. Tariff Revenue (B)') # Less interesting as only non-zero in tariff
plot_histograms('avg_elec_price_A', 'Avg. Electronics Price (in A)')
plot_histograms('avg_agri_price_B', 'Avg. Agri. Price (in B)')


# %% [markdown]
# ## Cell 6: Interpretation and Advice (Placeholder)
#
# **Executive Summary for Country A Policymakers:**
#
# Our simulation analysis compared a Free Trade policy for Electronics imports against imposing a 15% tariff, which triggered a 10% retaliatory tariff from Country B on our Agricultural exports.
#
# **Key Findings:**
#
# *   **Electronics Imports:** The tariff significantly reduced the volume of Electronics imports (Mean: [Insert Tariff Mean Elec Vol] vs. [Insert FT Mean Elec Vol] under free trade). This reduction is amplified by the availability of domestic substitutes, which become more attractive as import prices rise. The average price paid for imported Electronics increased substantially under the tariff scenario (Mean: [Insert Tariff Mean Elec Price] vs. [Insert FT Mean Elec Price]).
# *   **Agricultural Exports:** Our Agricultural exports were negatively impacted by the retaliatory tariff, showing a clear decrease in volume (Mean: [Insert Tariff Mean Agri Vol] vs. [Insert FT Mean Agri Vol]). The average price faced by Country B consumers for our Agricultural goods also increased (Mean: [Insert Tariff Mean Agri Price] vs. [Insert FT Mean Agri Price]).
# *   **Tariff Revenue:** While the tariff on Electronics generated revenue for Country A (Mean: [Insert Tariff Mean Elec Rev A]), this gain must be weighed against the losses in export volume. (Note: The retaliatory tariff generates revenue for Country B, not A).
# *   **Uncertainty:** Both scenarios exhibit considerable outcome variability due to exchange rate and demand fluctuations. The confidence intervals indicate the likely range for these outcomes. For instance, while the average Electronics tariff revenue is [X], there's a 95% chance the actual outcome over 5 years lies between [Y] and [Z]. Similarly, the damage to agricultural exports has a range of potential severity.
#
# **Advice:**
#
# Imposing the Electronics tariff offers the benefit of [Potential benefit, e.g., protecting domestic producers, generating revenue]. However, this comes at the direct cost of significantly reduced Electronics imports, higher prices for consumers/industries using them, and a quantifiable negative impact on our Agricultural export sector due to retaliation. The simulation highlights that the net effect on Country A's overall economic welfare is likely negative when retaliation is considered, as the losses in the export sector potentially outweigh the gains from the initial tariff. Policymakers must carefully weigh the desired protection for the domestic Electronics sector against the demonstrable harm to the Agricultural export market and the increased costs for Electronics consumers. The variability shown in the results also suggests preparing for a range of potential outcomes rather than relying solely on average predictions.

# %%