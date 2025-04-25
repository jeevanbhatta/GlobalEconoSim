import numpy as np
import matplotlib.pyplot as plt

# simulate tariff‐rate draws for a bunch of “countries”
n_countries = 80
samples1 = np.random.normal(loc=0.0, scale=0.15, size=n_countries)
samples2 = np.random.normal(loc=0.25, scale=0.15, size=n_countries)

# enforce a cutoff at 0
samples1 = np.clip(samples1, 0, None)
samples2 = np.clip(samples2, 0, None)

# plot histograms side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].hist(samples1, bins=30, color="C0", edgecolor="k")
axes[0].set_title("N(0.0, 0.15) clipped at 0")
axes[0].set_xlabel("Tariff rate")
axes[0].set_ylabel("Countries")

axes[1].hist(samples2, bins=30, color="C1", edgecolor="k")
axes[1].set_title("N(0.25, 0.15) clipped at 0")
axes[1].set_xlabel("Tariff rate")

plt.tight_layout()
plt.show()