import random
import networkx as nx

# Mesa core components
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

# --------------------------------------------------
# CUSTOM RANDOM SCHEDULER
# (Replacing mesa.schedule.RandomActivation)
# --------------------------------------------------
class SimpleRandomScheduler:
    """
    A scheduler that activates each agent once per step, in random order.
    This replicates the essential functionality of mesa.schedule.RandomActivation,
    since your Mesa version does not include the 'schedule' submodule.
    """
    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.time = 0
        # Keep track of agents in a dict keyed by unique_id
        self._agents = {}

    def add(self, agent):
        """
        Add an Agent object to the schedule.
        """
        self._agents[agent.unique_id] = agent

    def remove(self, agent):
        """
        Remove the specified Agent from the schedule.
        """
        if agent.unique_id in self._agents:
            del self._agents[agent.unique_id]

    def step(self):
        """
        Shuffle order and then activate each agent.
        """
        agent_keys = list(self._agents.keys())
        random.shuffle(agent_keys)
        for key in agent_keys:
            self._agents[key].step()
        self.steps += 1
        self.time += 1


# --------------------------------------------------
# 1. MICRO-LEVEL AGENT
# --------------------------------------------------
class MicroAgent(Agent):
    """
    Represents an individual/household within a country (MacroAgent).
    Includes mechanics for employment, resource accumulation, R&D,
    migration, etc.
    """
    def __init__(
        self, 
        unique_id, 
        model, 
        country_id,
        gender, 
        race, 
        education, 
        employment, 
        resources, 
        innovation, 
        cooperation, 
        trade_propensity, 
        vulnerability
    ):
        # According to your error, Agent requires Agent.__init__(self, model).
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.model = model
        
        # Link to the macro-level (country) agent
        self.country_id = country_id
        
        # Demographics
        self.gender = gender
        self.race = race
        
        # Human capital & employment
        self.education = education
        self.employment = employment  # "employed" or "unemployed"
        
        # Economic attributes
        self.resources = resources
        self.innovation = innovation
        self.cooperation = cooperation
        self.trade_propensity = trade_propensity
        
        # Conflict / shock vulnerability
        self.vulnerability = vulnerability

    def step(self):
        """
        Main decision loop for a micro agent:
          1. Possibly migrate if conditions are poor
          2. Update or find employment
          3. Make economic decisions (invest in R&D or education, trade, cooperate)
        """
        self.possibly_migrate()
        self.update_employment()
        self.make_economic_decision()

    def possibly_migrate(self):
        """
        Example heuristic:
        - If agent's resources are below a threshold and 
          conflict is high in its current country, there's a chance to move.
        """
        country = self.model.get_country_agent(self.country_id)
        if country is None:
            return
        
        conflict_level = country.conflict_initiation_chance  # a stand-in for 'instability'
        if self.resources < 30 and random.random() < conflict_level * 0.5:
            possible_countries = [
                m for m in self.model.macro_agents
                if m.unique_id != self.country_id
            ]
            if possible_countries:
                new_country = random.choice(possible_countries)
                self.country_id = new_country.unique_id

    def update_employment(self):
        """
        Probability of gaining or losing employment.
        Influenced by education and country-level policy.
        """
        country = self.model.get_country_agent(self.country_id)
        if not country:
            return
        
        base_prob = 0.1 + 0.4 * self.education
        base_prob += country.employment_policy_boost
        
        if self.employment == "unemployed":
            if random.random() < base_prob:
                self.employment = "employed"
        else:
            # Optionally, there's a small chance to lose employment
            if random.random() < 0.02:  # e.g., 2% layoff
                self.employment = "unemployed"

    def make_economic_decision(self):
        """
        Decide how to allocate resources.
        Might invest in innovation, education, or attempt trade/cooperation.
        """
        if self.resources > 50:
            if random.random() < 0.5:
                self.invest_in_innovation()
            else:
                self.improve_education()
        else:
            self.trade_or_cooperate()

    def invest_in_innovation(self):
        investment = 10
        if self.resources >= investment:
            self.resources -= investment
            country = self.model.get_country_agent(self.country_id)
            rd_policy = country.rd_policy_boost if country else 0.0
            self.innovation += investment * self.model.innovation_factor * (
                1.0 + self.education + rd_policy
            )

    def improve_education(self):
        cost = 8
        if self.resources >= cost and self.education < 1.0:
            self.resources -= cost
            country = self.model.get_country_agent(self.country_id)
            edu_subsidy = country.education_subsidy if country else 0.0
            self.education += 0.05 * (1.0 + edu_subsidy)
            self.education = min(self.education, 1.0)

    def trade_or_cooperate(self):
        neighbors = self.model.get_neighbors(self)
        if not neighbors:
            return
        partner = self.random.choice(neighbors)
        if random.random() < self.trade_propensity:
            self.execute_trade(partner)
        else:
            self.execute_cooperation(partner)

    def execute_trade(self, partner):
        """
        Basic bilateral trade model:
         - Each invests some resources
         - Gains synergy
         - Potentially exchange innovation
        """
        trade_amount = 5
        if (
            self.resources >= trade_amount and
            partner.resources >= trade_amount and
            random.random() < partner.trade_propensity
        ):
            self.resources -= trade_amount
            partner.resources -= trade_amount
            synergy_gain = 8
            self.resources += synergy_gain / 2
            partner.resources += synergy_gain / 2
            
            diff = partner.innovation - self.innovation
            self.innovation += diff * 0.05
            partner.innovation -= diff * 0.05

    def execute_cooperation(self, partner):
        """
        Agents cooperate: share some resources, average out innovation.
        """
        if (
            random.random() < self.cooperation and
            random.random() < partner.cooperation
        ):
            resource_transfer = 3
            if self.resources > resource_transfer:
                self.resources -= resource_transfer
                partner.resources += resource_transfer
            
            avg_innovation = (self.innovation + partner.innovation) / 2
            self.innovation = avg_innovation
            partner.innovation = avg_innovation


# --------------------------------------------------
# 2. MACRO-LEVEL AGENT (COUNTRY)
# --------------------------------------------------
class MacroAgent(Agent):
    """
    Represents a country or region. Holds policy levers like 
    tax rates, R&D support, conflict parameters, etc.
    """
    def __init__(
        self, 
        unique_id, 
        model, 
        name,
        employment_policy_boost, 
        education_subsidy, 
        rd_policy_boost, 
        conflict_initiation_chance, 
        tax_rate
    ):
        # Agent requires Agent.__init__(self, model).
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.model = model
        
        self.name = name
        self.employment_policy_boost = employment_policy_boost
        self.education_subsidy = education_subsidy
        self.rd_policy_boost = rd_policy_boost
        self.conflict_initiation_chance = conflict_initiation_chance
        self.tax_rate = tax_rate  # e.g., 0.0 to 0.5

    def step(self):
        """
        Each step:
          1. Possibly initiate conflict
          2. Apply taxes and redistribute (subsidies) to micro agents
        """
        self.possibly_initiate_conflict()
        self.apply_taxes_and_subsidies()

    def possibly_initiate_conflict(self):
        """
        With a given probability, conflict occurs in this country,
        reducing the resources of micro agents.
        """
        if random.random() < self.conflict_initiation_chance:
            self.apply_conflict()

    def apply_conflict(self):
        micro_agents = self.model.get_micro_agents_in_country(self.unique_id)
        for agent in micro_agents:
            loss_fraction = random.uniform(0.1, 0.3)
            loss_amount = agent.resources * agent.vulnerability * loss_fraction
            agent.resources = max(0, agent.resources - loss_amount)
            
            if random.random() < 0.5:
                agent.innovation = max(0, agent.innovation - 0.1)
            if random.random() < 0.1:
                agent.employment = "unemployed"

    def apply_taxes_and_subsidies(self):
        """
        Simple model:
         - Collect a fraction of each agent's resources as tax.
         - Redistribute total as direct subsidy to micro agents.
        """
        micro_agents = self.model.get_micro_agents_in_country(self.unique_id)
        if not micro_agents:
            return
        
        total_taxes = 0
        for agent in micro_agents:
            tax = agent.resources * self.tax_rate
            agent.resources -= tax
            total_taxes += tax
        
        if total_taxes > 0:
            subsidy_per_agent = total_taxes / len(micro_agents)
            for agent in micro_agents:
                agent.resources += subsidy_per_agent


# --------------------------------------------------
# 3. MAIN MODEL
# --------------------------------------------------
class GlobalDevelopmentModel(Model):
    """
    Contains both macro (country) agents and micro (individual) agents.
    Sets up the network environment, runs the scheduler, and collects data.
    """
    def __init__(
        self, 
        num_countries=2, 
        agents_per_country=50,
        innovation_factor=0.02
    ):
        super().__init__()
        
        self.num_countries = num_countries
        self.agents_per_country = agents_per_country
        self.innovation_factor = innovation_factor
        
        # A custom counter for generating unique IDs (since next_id isn't available).
        self.current_id = 0
        
        # REPLACE the Mesa scheduler with our custom scheduler
        self.schedule = SimpleRandomScheduler(self)
        
        # Network for micro-level interactions
        self.G = nx.Graph()
        self.grid = NetworkGrid(self.G)

        # 1. Create Macro (Country) Agents
        self.macro_agents = []
        for c_id in range(num_countries):
            name = f"Country-{c_id}"
            emp_boost = random.uniform(0.0, 0.3)
            edu_subsidy = random.uniform(0.0, 0.3)
            rd_boost = random.uniform(0.0, 0.2)
            conflict_chance = random.uniform(0.0, 0.05)
            tax_rate = random.uniform(0.0, 0.3)
            
            macro_agent = MacroAgent(
                unique_id=self.next_id(),
                model=self,
                name=name,
                employment_policy_boost=emp_boost,
                education_subsidy=edu_subsidy,
                rd_policy_boost=rd_boost,
                conflict_initiation_chance=conflict_chance,
                tax_rate=tax_rate
            )
            self.schedule.add(macro_agent)
            self.macro_agents.append(macro_agent)

        # 2. Create Micro Agents in each Country
        self.micro_agents = []
        for macro_agent in self.macro_agents:
            for _ in range(agents_per_country):
                micro_agent = MicroAgent(
                    unique_id=self.next_id(),
                    model=self,
                    country_id=macro_agent.unique_id,
                    gender=random.choice(["male", "female"]),
                    race=random.choice(["GroupA", "GroupB"]),
                    education=random.uniform(0, 1),
                    employment=random.choice(["employed", "unemployed"]),
                    resources=random.randint(50, 150),
                    innovation=random.uniform(0, 0.5),
                    cooperation=random.uniform(0.3, 0.7),
                    trade_propensity=random.uniform(0.2, 0.6),
                    vulnerability=random.uniform(0.1, 0.3)
                )
                self.schedule.add(micro_agent)
                self.micro_agents.append(micro_agent)
                self.G.add_node(micro_agent.unique_id)

        # 3. Connect micro agents into networks
        self.connect_agents()

        # 4. Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Global_Avg_Innovation": self.compute_global_avg_innovation,
                "Global_Avg_Resources": self.compute_global_avg_resources,
                "Global_Employment_Rate": self.compute_global_employment_rate,
                "Global_Gini_Resources": self.compute_global_gini_resources
            },
            agent_reporters={
                # Macro-level
                "Macro_Name": lambda a: a.name if isinstance(a, MacroAgent) else None,
                "Macro_TaxRate": lambda a: a.tax_rate if isinstance(a, MacroAgent) else None,
                # Micro-level
                "Micro_CountryID": lambda a: a.country_id if isinstance(a, MicroAgent) else None,
                "Micro_Resources": lambda a: a.resources if isinstance(a, MicroAgent) else None,
                "Micro_Innovation": lambda a: a.innovation if isinstance(a, MicroAgent) else None,
                "Micro_Education": lambda a: a.education if isinstance(a, MicroAgent) else None,
            }
        )

    def next_id(self):
        """
        Increment and return a unique integer ID for agents.
        """
        self.current_id += 1
        return self.current_id

    def connect_agents(self):
        """
        Simple function to create edges among micro agents, primarily within each country,
        plus some cross-country links.
        """
        from collections import defaultdict
        country_groups = defaultdict(list)
        for ma in self.micro_agents:
            country_groups[ma.country_id].append(ma)

        # Connect micro agents within each country
        for c_id, agents_list in country_groups.items():
            for i in range(len(agents_list) - 1):
                if random.random() < 0.7:
                    a1 = agents_list[i]
                    a2 = agents_list[i+1]
                    self.G.add_edge(a1.unique_id, a2.unique_id)

        # Random cross-country connections
        if self.num_countries > 1:
            for _ in range(10):
                a1 = random.choice(self.micro_agents)
                a2 = random.choice(self.micro_agents)
                if a1.country_id != a2.country_id:
                    self.G.add_edge(a1.unique_id, a2.unique_id)

    def get_neighbors(self, agent):
        """
        Return micro-agents connected to the given micro-agent.
        """
        node_id = agent.unique_id
        neighbors = []
        if node_id in self.G.nodes:
            for nid in self.G.neighbors(node_id):
                neighbor_agent = self.schedule._agents[nid]
                neighbors.append(neighbor_agent)
        return neighbors

    def get_country_agent(self, country_id):
        """
        Retrieve the macro-level (country) agent by its unique_id.
        """
        return self.schedule._agents.get(country_id, None)

    def get_micro_agents_in_country(self, country_id):
        """
        Return all micro-agents that belong to a specific country.
        """
        return [a for a in self.micro_agents if a.country_id == country_id]

    # -----------------------------------
    # Data Collection Functions
    # -----------------------------------
    def compute_global_avg_innovation(self):
        if not self.micro_agents:
            return 0
        return sum(a.innovation for a in self.micro_agents) / len(self.micro_agents)

    def compute_global_avg_resources(self):
        if not self.micro_agents:
            return 0
        return sum(a.resources for a in self.micro_agents) / len(self.micro_agents)

    def compute_global_employment_rate(self):
        if not self.micro_agents:
            return 0
        employed = sum(1 for a in self.micro_agents if a.employment == "employed")
        return employed / len(self.micro_agents)

    def compute_global_gini_resources(self):
        """
        Simple Gini coefficient calculation for resource distribution.
        """
        resources = sorted(a.resources for a in self.micro_agents)
        if not resources:
            return 0
        n = len(resources)
        cum_resources = 0
        cum_weighted = 0
        total_resources = sum(resources)
        for i, r in enumerate(resources):
            cum_resources += r
            cum_weighted += (i+1)*r
        gini = (2 * cum_weighted) / (n * total_resources) - (n+1)/n
        return gini

    def step(self):
        """
        Each step, all agents (macro + micro) act, then data is collected.
        """
        self.schedule.step()
        self.datacollector.collect(self)


# --------------------------------------------------
# 4. OPTIONAL: STANDALONE RUN (no visualization)
# --------------------------------------------------
if __name__ == "__main__":
    model = GlobalDevelopmentModel(num_countries=3, agents_per_country=40, innovation_factor=0.03)
    for step in range(100):
        model.step()

    df_model = model.datacollector.get_model_vars_dataframe()
    print("Global Indicators (last 5 steps):\n", df_model.tail())
    
    df_agents = model.datacollector.get_agent_vars_dataframe()
    print("\nSample of Agent-Level Data:\n", df_agents.head(20))
