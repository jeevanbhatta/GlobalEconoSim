import numpy as np
import random
import itertools
from dataclasses import dataclass, field
from typing import Dict, List
import networkx as nx

@dataclass
class Country:
    cid: int
    bloc: int
    gdp: float
    population: int
    poverty_rate: float
    dev_index: float
    eff: Dict[str, float]
    exports: float = 0.0
    imports: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "gdp": [],
        "poverty_rate": [],
        "exports": [],
        "imports": [],
    })

    def log_step(self):
        for k in self.history:
            self.history[k].append(getattr(self, k))

class TradeNetwork:
    def __init__(self, countries, conn_intra, conn_inter, tariff_intra_mu, tariff_inter_mu, tariff_sd):
        self.countries = {c.cid: c for c in countries}
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.countries.keys())
        n = len(countries)
        for i, j in itertools.permutations(range(n), 2):
            same_bloc = countries[i].bloc == countries[j].bloc
            p = conn_intra if same_bloc else conn_inter
            if random.random() < p:
                mu = tariff_intra_mu if same_bloc else tariff_inter_mu
                tariff = np.clip(np.random.normal(mu, tariff_sd), 0, 1)
                dist = random.uniform(1, 10)
                self.G.add_edge(i, j, tariff=tariff, dist=dist, reciprocal=False)
        
        # Track history of tariff changes for diplomatic analysis
        self.tariff_history = {}
        self.diplomatic_relations = {}

    def compute_trade_flows(self, goods: List[str], price_elast: float = 1.0):
        for u, v, data in self.G.edges(data=True):
            c_u, c_v = self.countries[u], self.countries[v]
            tariff, dist = data["tariff"], data["dist"]
            total_flow = 0.0
            for g in goods:
                prod = c_u.eff[g]
                # Scale down GDP values to prevent overflow
                gdp_u_scaled = c_u.gdp / 1e6  # Scale down by a million
                gdp_v_scaled = c_v.gdp / 1e6  # Scale down by a million
                try:
                    # Use np.clip to limit flow to reasonable bounds
                    flow = np.clip((prod * gdp_u_scaled * gdp_v_scaled) / (dist ** 2), 0, 1e12)
                    flow *= (1 - tariff) ** price_elast
                    # Check for NaN or inf values
                    if np.isfinite(flow):
                        total_flow += flow
                    else:
                        flow = 0
                except (OverflowError, FloatingPointError):
                    flow = 0
            data["flow"] = np.clip(total_flow, 0, 1e12)  # Ensure flow is finite and non-negative
        
        for c in self.countries.values():
            outs = self.G.out_edges(c.cid, data=True)
            ins = self.G.in_edges(c.cid, data=True)
            try:
                c.exports = sum(d.get("flow", 0) for *_, d in outs)
                c.imports = sum(d.get("flow", 0) for *_, d in ins)
            except (OverflowError, FloatingPointError):
                # Handle error by setting to safe values
                c.exports = 0.0
                c.imports = 0.0

    def apply_tariff_delta(self, cid: int, delta: float):
        # Apply tariff to all outgoing edges from the country
        for _, v, d in self.G.out_edges(cid, data=True):
            old_tariff = d["tariff"]
            d["tariff"] = np.clip(old_tariff + delta, 0, 1)
            # Mark edge as being under a reciprocal tariff regime
            d["reciprocal"] = True
            
            # Record this tariff change in history
            edge_key = f"{cid}-{v}"
            if edge_key not in self.tariff_history:
                self.tariff_history[edge_key] = []
            self.tariff_history[edge_key].append((old_tariff, d["tariff"]))
            
            # Update diplomatic relations score
            if cid not in self.diplomatic_relations:
                self.diplomatic_relations[cid] = {}
            if v not in self.diplomatic_relations[cid]:
                self.diplomatic_relations[cid][v] = 1.0  # Start with neutral relations
            
            # Relations worsen with tariff increases (delta > 0) and improve with decreases (delta < 0)
            self.diplomatic_relations[cid][v] = np.clip(self.diplomatic_relations[cid][v] - delta, 0, 1)
            
            # Add reciprocal tariffs from other countries
            # Other countries will respond with a partial reciprocal tariff (70% of original)
            if self.G.has_edge(v, cid):
                recip_edge = self.G[v][cid]
                old_recip_tariff = recip_edge["tariff"]
                recip_delta = delta * 0.7  # 70% reciprocal response
                recip_edge["tariff"] = np.clip(recip_edge["tariff"] + recip_delta, 0, 1)
                recip_edge["reciprocal"] = True
                
                # Record reciprocal tariff change in history
                recip_edge_key = f"{v}-{cid}"
                if recip_edge_key not in self.tariff_history:
                    self.tariff_history[recip_edge_key] = []
                self.tariff_history[recip_edge_key].append((old_recip_tariff, recip_edge["tariff"]))
                
                # Update reciprocal diplomatic relations
                if v not in self.diplomatic_relations:
                    self.diplomatic_relations[v] = {}
                if cid not in self.diplomatic_relations[v]:
                    self.diplomatic_relations[v][cid] = 1.0
                
                # Relations worsen with tariff increases
                self.diplomatic_relations[v][cid] = np.clip(self.diplomatic_relations[v][cid] - recip_delta, 0, 1)
    
    def get_country_friendship_stats(self, cid: int):
        """
        Returns average tariff data for incoming and outgoing connections of a country
        """
        out_edges = list(self.G.out_edges(cid, data=True))
        in_edges = list(self.G.in_edges(cid, data=True))
        
        outgoing_tariffs = [data['tariff'] for _, _, data in out_edges]
        incoming_tariffs = [data['tariff'] for _, _, data in in_edges]
        
        out_avg = sum(outgoing_tariffs) / len(outgoing_tariffs) if outgoing_tariffs else 0
        in_avg = sum(incoming_tariffs) / len(incoming_tariffs) if incoming_tariffs else 0
        
        # Count countries in same bloc
        same_bloc_count = sum(1 for _, c in self.countries.items() 
                            if c.bloc == self.countries[cid].bloc and c.cid != cid)
        
        return {
            "outgoing_avg_tariff": out_avg,
            "incoming_avg_tariff": in_avg,
            "same_bloc_count": same_bloc_count,
            "outgoing_tariffs": outgoing_tariffs,
            "incoming_tariffs": incoming_tariffs,
            "trading_partners": len(outgoing_tariffs),
        }
        
    def allow_bloc_formation(self, tariff_threshold=0.4):
        """
        Form new blocs in response to high tariffs.
        Countries with low mutual tariffs might form a new bloc.
        """
        n_countries = len(self.countries)
        # Find pairs of countries with low mutual tariffs
        potential_bloc_pairs = []
        
        for i, j in itertools.combinations(range(n_countries), 2):
            # Skip if already in same bloc
            if self.countries[i].bloc == self.countries[j].bloc:
                continue
                
            # Check tariffs both ways if the edges exist
            i_to_j_tariff = self.G[i][j]['tariff'] if self.G.has_edge(i, j) else 1.0
            j_to_i_tariff = self.G[j][i]['tariff'] if self.G.has_edge(j, i) else 1.0
            
            # If both tariffs are below threshold, consider forming a bloc
            if i_to_j_tariff < tariff_threshold and j_to_i_tariff < tariff_threshold:
                avg_tariff = (i_to_j_tariff + j_to_i_tariff) / 2
                potential_bloc_pairs.append((i, j, avg_tariff))
        
        # Sort by lowest average tariff
        potential_bloc_pairs.sort(key=lambda x: x[2])
        
        # Assign new bloc IDs starting from the highest existing bloc ID + 1
        max_bloc = max(country.bloc for country in self.countries.values()) + 1
        new_blocs = {}
        
        # Try to form new blocs
        for i, j, _ in potential_bloc_pairs:
            # If either country is already assigned to a new bloc, add the other to that bloc
            if i in new_blocs and j not in new_blocs:
                new_blocs[j] = new_blocs[i]
            elif j in new_blocs and i not in new_blocs:
                new_blocs[i] = new_blocs[j]
            # If neither is assigned, create a new bloc
            elif i not in new_blocs and j not in new_blocs:
                new_blocs[i] = max_bloc
                new_blocs[j] = max_bloc
                max_bloc += 1
        
        # Apply new bloc assignments
        for cid, bloc in new_blocs.items():
            self.countries[cid].bloc = bloc
        
        return len(new_blocs) > 0  # Return True if any new blocs were formed

    def get_diplomatic_relations_network(self):
        """
        Returns a network representation of diplomatic relations between countries
        """
        relations_G = nx.DiGraph()
        relations_G.add_nodes_from(self.countries.keys())
        
        for source, targets in self.diplomatic_relations.items():
            for target, relation in targets.items():
                relations_G.add_edge(source, target, weight=relation)
        
        return relations_G

def simulate(n, blocs=0, steps=100, conn_intra=0.7, conn_inter=0.3, tariff_gap=0.1, tariff_sd=0.05, two_goods=False, policy_shock=None, additional_shocks=None):
    goods = ["A", "B"] if two_goods else ["A"]
    countries = []
    for cid in range(n):
        bloc = cid % max(1, blocs)  # Ensure valid bloc assignment even when blocs=0
        gdp0 = random.uniform(10000, 100000)
        pop = random.randint(2_000_000, 150_000_000)
        pov = random.uniform(0.05, 0.5)
        dev = random.uniform(0.4, 0.9)
        eff = {g: abs(np.random.normal(1.0, 0.3)) for g in goods}
        countries.append(Country(cid, bloc, gdp0, pop, pov, dev, eff))
    
    # When blocs=0, treat all connections the same
    effective_conn_intra = conn_intra if blocs > 0 else (conn_intra + conn_inter) / 2
    effective_conn_inter = conn_inter if blocs > 0 else effective_conn_intra
    
    effective_tariff_intra_mu = max(0.05, 0.10 - tariff_gap / 2) if blocs > 0 else 0.10
    effective_tariff_inter_mu = min(0.95, 0.10 + tariff_gap / 2) if blocs > 0 else 0.10
    
    net = TradeNetwork(
        countries,
        effective_conn_intra,
        effective_conn_inter,
        tariff_intra_mu=effective_tariff_intra_mu,
        tariff_inter_mu=effective_tariff_inter_mu,
        tariff_sd=tariff_sd,
    )
    
    # Track friendship stats history for potential shock country
    friendship_stats_history = {}
    if policy_shock is not None and isinstance(policy_shock, tuple) and len(policy_shock) == 2:
        shock_id, _ = policy_shock
        if shock_id is not None and isinstance(shock_id, int) and 0 <= shock_id < n:
            friendship_stats_history = {
                "time": [],
                "outgoing_avg_tariff": [],
                "incoming_avg_tariff": [],
                "same_bloc_count": [],
                "trading_partners": [],
                "shock_points": []  # Will track all shock points
            }
    
    # Initialize additional_shocks if not provided
    if additional_shocks is None:
        additional_shocks = []
    
    # Convert additional_shocks to a list of (t, shock_id, shock_delta) tuples if not already
    formatted_additional_shocks = []
    for shock in additional_shocks:
        if isinstance(shock, tuple):
            if len(shock) == 3:  # Already in format (t, shock_id, shock_delta)
                formatted_additional_shocks.append(shock)
            elif len(shock) == 2:  # In format (shock_id, shock_delta), need to add time
                # Default to halfway through remaining simulation
                t = steps // 2
                formatted_additional_shocks.append((t, shock[0], shock[1]))
    
    # Combine initial shock and additional shocks for processing
    all_shocks = []
    if policy_shock is not None and isinstance(policy_shock, tuple) and len(policy_shock) == 2:
        # Initial shock at 1/4 through simulation
        all_shocks.append((steps // 4, policy_shock[0], policy_shock[1]))
    
    # Add all additional shocks
    all_shocks.extend(formatted_additional_shocks)
    
    # Sort shocks by time
    all_shocks.sort(key=lambda x: x[0])
    
    for t in range(steps):
        # Check for valid policy shocks at this time step
        shock_applied = False
        
        # Process any shocks scheduled for this time step
        for shock_time, shock_id, shock_delta in all_shocks:
            if t == shock_time and 0 <= shock_id < n:
                try:
                    net.apply_tariff_delta(shock_id, shock_delta)
                    # Try to form new blocs in response to tariff changes
                    net.allow_bloc_formation(tariff_threshold=0.4)
                    shock_applied = True
                    
                    # Mark the shock point in the history
                    if friendship_stats_history:
                        friendship_stats_history["shock_points"].append(t)
                except (TypeError, ValueError):
                    # If shock parameters are invalid, just skip applying the shock
                    pass
        
        # Consider forming new blocs every 10 steps in response to changing trade patterns
        elif t > 0 and t % 10 == 0:
            net.allow_bloc_formation(tariff_threshold=0.4)
        
        net.compute_trade_flows(goods)
        for c in countries:
            # Handle exports and imports safely
            exports = min(c.exports, 1e12)  # Cap exports at reasonable value
            imports = min(c.imports, 1e12)  # Cap imports at reasonable value
            
            # Calculate trade balance with safeguards
            trade_balance = 0.3 * exports - 0.2 * imports
            
            # Limit GDP growth to prevent exponential explosion
            # Use tanh to limit growth between -0.3 and +0.3 (reasonable economic bounds)
            raw_growth = trade_balance / max(c.gdp, 1e3)
            bounded_growth = 0.02 + 0.3 * np.tanh(raw_growth)  # Base growth + bounded trade effect
            
            # Apply GDP growth with safety checks
            c.gdp = min(c.gdp * (1 + bounded_growth), 1e15)  # Cap GDP at reasonable maximum
            
            # Update poverty rate safely
            try:
                growth_effect = np.clip(bounded_growth - 0.01, -0.5, 0.5)  # Limit effect
                c.poverty_rate = max(c.poverty_rate * (1 - 0.3 * growth_effect), 0.01)
            except (OverflowError, FloatingPointError):
                # If calculation fails, apply a safe default poverty reduction
                c.poverty_rate = max(c.poverty_rate * 0.99, 0.01)
            
            c.log_step()
        
        # Track friendship stats for shocked country
        if policy_shock is not None and isinstance(policy_shock, tuple) and len(policy_shock) == 2:
            shock_id, _ = policy_shock
            if shock_id is not None and isinstance(shock_id, int) and 0 <= shock_id < n:
                stats = net.get_country_friendship_stats(shock_id)
                friendship_stats_history["time"].append(t)
                friendship_stats_history["outgoing_avg_tariff"].append(stats["outgoing_avg_tariff"])
                friendship_stats_history["incoming_avg_tariff"].append(stats["incoming_avg_tariff"])
                friendship_stats_history["same_bloc_count"].append(stats["same_bloc_count"])
                friendship_stats_history["trading_partners"].append(stats["trading_partners"])
    
    # Store the friendship stats history in the network object
    net.friendship_stats_history = friendship_stats_history
    
    return net