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
                self.G.add_edge(i, j, tariff=tariff, dist=dist)

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
        for _, v, d in self.G.out_edges(cid, data=True):
            d["tariff"] = np.clip(d["tariff"] + delta, 0, 1)

def simulate(n, blocs, steps, conn_intra, conn_inter, tariff_gap, tariff_sd, two_goods, policy_shock=None):
    goods = ["A", "B"] if two_goods else ["A"]
    countries = []
    for cid in range(n):
        bloc = cid % blocs
        gdp0 = random.uniform(10000, 100000)
        pop = random.randint(2_000_000, 150_000_000)
        pov = random.uniform(0.05, 0.5)
        dev = random.uniform(0.4, 0.9)
        eff = {g: abs(np.random.normal(1.0, 0.3)) for g in goods}
        countries.append(Country(cid, bloc, gdp0, pop, pov, dev, eff))
    net = TradeNetwork(
        countries,
        conn_intra,
        conn_inter,
        tariff_intra_mu=max(0.05, 0.10 - tariff_gap / 2),
        tariff_inter_mu=min(0.95, 0.10 + tariff_gap / 2),
        tariff_sd=tariff_sd,
    )
    for t in range(steps):
        if policy_shock and t == steps // 4:
            net.apply_tariff_delta(*policy_shock)
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
    return net