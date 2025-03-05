## GlobalEconoSim

A simulation of global economic development that models both **MicroAgents** (individuals/households) and **MacroAgents** (countries/regions). Each step, agents make decisions, interact, and update their states. The simulation tracks key indicators over time (e.g., average resources, Gini coefficient).

---

### 1. MicroAgents (Individuals or Households)

- **Core Attributes**:
  - **Education**: Ranges from 0 to 1 (continuous measure).
  - **Employment**: "employed" or "unemployed."
  - **Resources**: Numerical wealth or savings.
  - **Innovation**: Measure of innovative capacity or IP.
  - **Vulnerability**: How strongly conflicts/shocks affect this agent.
  - **Demographics**: **Gender**, **race**, **age**, **skill_level**, **health_condition**, etc.

- **Behaviors**:
  1. **Migrate** if resources are too low and conflict risk is high.
  2. **Update Employment**: Seek or lose jobs based on education, skill level, and country policies.
  3. **Make Economic Decisions**:
     - **Invest in R&D** (innovation) or **improve education**, if resources allow.
     - **Trade or Cooperate** with neighboring agents in the network.
     - **Consume** daily resources (simple consumption model).

---

### 2. MacroAgents (Countries or Regions)

- **Core Attributes**:
  - **Name/ID**: Unique identifier (e.g., "Country-0").
  - **Policy Levers**:
    - **Tax Rate**  
    - **Employment Policy Boost**  
    - **Education Subsidy**  
    - **R&D Policy Boost**
  - **Conflict Parameters**: Probability of triggering conflict, which reduces micro agents’ resources.
  - **GDP**, **inflation_rate**, **currency_exchange_rate**, **environment_policy**, **trade_policy** (basic placeholders to demonstrate macroeconomic factors).

- **Behaviors**:
  1. **Initiate Conflict**: With some probability, conflict breaks out and harms local agents’ resources.
  2. **Taxes & Subsidies**: Collect a fraction of each MicroAgent’s resources, then redistribute as subsidies.
  3. **Update Macro Indicators**: Simple random fluctuations in GDP, inflation, etc., to simulate macroeconomic dynamics.

---

### 3. GlobalDevelopmentModel

- **Initialization**:
  1. Creates `num_countries` **MacroAgents** with random policy parameters.
  2. For each MacroAgent, creates `agents_per_country` **MicroAgents** with random demographics (gender, race, age, etc.) and random initial resources.
  3. Builds a **network graph** (`networkx.Graph`) connecting MicroAgents. Also allows random cross-country connections.

- **Step Flow**:
  1. **MacroAgents** act first:
     - Possibly initiate conflict.
     - Collect taxes, redistribute subsidies.
     - Update macro indicators (GDP, inflation, etc.).
  2. **MicroAgents** act:
     - Possibly migrate if conditions are poor.
     - Update or lose employment based on skill level, education, and policy.
     - Make economic decisions: invest in innovation or education, or trade/cooperate with neighbors.
  
- **Data Collection**:
  - **Model-Level**: Average innovation, average resources, global employment rate, Gini coefficient.  
  - **Agent-Level**: Macro agent attributes (GDP, inflation, tax rate, etc.) and Micro agent attributes (gender, race, skill, health, resources, etc.).

---

### 4. Visualization

Depending on your environment and Mesa version, you can visualize:
1. **Network Diagram** of MicroAgents (often circles) and optional MacroAgents (squares).  
2. **Charts** for global indicators:  
   - Average Innovation  
   - Average Resources  
   - Employment Rate  
   - Gini Coefficient  

---

### 5. Usage

Below are two Python files:

- **`model.py`**  
  Contains all the agent classes (`MicroAgent`, `MacroAgent`) plus the main `GlobalDevelopmentModel` logic.

- **`server.py`**  
  Sets up an **interactive** interface for the model. Depending on your Mesa version:  
  - You might use **`ModularServer`** to show a built-in Mesa interface.  
  - Or you can adopt **Streamlit**/**PyVis** for a custom UI.

#### Running the Model (No Visualization)

1. In a terminal:
   ```bash
   python model.py
   ```
   This will run the model for a fixed number of steps and print out final stats.

#### Interactive Visualization (Example Using Mesa’s ModularServer)

```bash
pip install -r requirements.txt
python streamlit_server.py
```
Then open the printed URL (by default `http://127.0.0.1:8521`) to see the network diagram and real-time charts.  

> **Note**: If your Mesa version doesn’t include `ModularVisualization`, you can run a custom **Streamlit** app or use any other visualization approach.

---

### 6. Extending the Simulation

- **Country Alliances**: Model alliances or trade deals between MacroAgents.  
- **Health/Education Shocks**: E.g., pandemics reducing productivity or innovation.  
- **Environmental Policy**: Let environment_policy reduce conflicts or daily consumption over time.  

With **GlobalEconoSim**, you can flexibly explore how micro-level behaviors (consumption, skill, migration) and macro-level policies (tax, conflict, R&D) interplay to shape development outcomes.