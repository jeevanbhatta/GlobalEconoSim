## GlobalEconoSim

1. **MicroAgents (Individuals or Households)**  
   - **Attributes**: Education, employment, resources (wealth), innovation, vulnerability to conflict, demographic factors (gender, race), etc.  
   - **Behaviors**:  
     - Invest in R&D or education  
     - Trade or cooperate with neighbors  
     - Possibly migrate if conflict is high or resources are too low  

2. **MacroAgents (Countries or Regions)**  
   - **Attributes**: Policy levers (tax rates, subsidies, R&D support), conflict tendencies, name/identifier, etc.  
   - **Behaviors**:  
     - Apply taxes and redistribute as subsidies  
     - Initiate or manage conflict  
     - Potentially sign alliances or trade deals (extension idea)

3. **GlobalDevelopmentModel**  
   - Creates both macro- and micro-level agents.  
   - Sets up a **network** or **graph** structure for micro-level interactions.  
   - Each step: 
     1. MacroAgents apply policies, possibly trigger conflict.  
     2. MicroAgents then update employment, invest or trade, possibly migrate.  
   - Collects data on global and country-level indicators (e.g., average resources, Gini index).  

4. **Visualization**  
   - **Network Visualization**: Each MicroAgent is a node in a network. MacroAgents can be displayed as separate nodes or abstracted.  
   - **Charts**: Track time series of global indicators such as average innovation, average resources, employment rate, and Gini coefficient.  
   - **Interactive Controls** (optional advanced usage): Let the user set certain parameters (e.g., number of steps, tax rate) at runtime in the Mesa GUI.

Below are **two Python files**:  
- **`model.py`**: Contains all agent classes and the main model logic.  
- **`server.py`**: Sets up the Mesa interactive server, including a **network visualization** and a set of **chart modules** to track key metrics in real time.

> **Instructions**  
> 1. In a terminal, run:  
>    ```bash
>    pip install mesa networkx
>    python server.py
>    ```  
> 2. Open the provided local URL (usually [http://127.0.0.1:8521](http://127.0.0.1:8521)) in your browser to see and interact with the simulation.
