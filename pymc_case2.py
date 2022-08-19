#!/usr/bin/env python3
# =============================================================================
#     File: pymc_case2.py
#  Created: 2019-06-20 23:24
#   Author: Bernie Roesler
#
"""
  Description: Case study of coal mine accident data.
  See: <https://www.kaggle.com/code/billbasener/switchpoint-analysis-of-mining-disasters-in-pymc3/notebook>
  for data.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

plt.style.use('seaborn-darkgrid')

disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                           3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                           2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
                           1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                           3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

years = np.arange(1851, 1962)

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.plot(years, disaster_data, 'o', markersize=8)
ax.set(ylabel='Disaster count',
       xlabel='Year')

with pm.Model() as disaster_model:
    # Switch point when disaster rate reduced, ensure test value in range
    s = pm.DiscreteUniform('s', lower=years.min(), upper=years.max(),
                           initval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(years <= s, early_rate, late_rate)

    # Define the number of disasters as a Poisson distribution
    disasters = pm.Poisson('disasters', rate, observed=disaster_data)

# Run the MCMC using Metropolis-Hastings algorithm (discrete variables)
Ns = 10000
with disaster_model:
    trace = pm.sample(Ns)

# az.plot_trace(trace)  # plot the convergence of parameters

fig = plt.figure(3, clear=True, constrained_layout=True)
fig.set_size_inches((8, 6), forward=True)
ax = fig.add_subplot()
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')

# Plot raw data
ax.scatter(years, disaster_data)

ax.axvline(trace.posterior.s.mean(),
           disaster_data.min(), disaster_data.max(), color='C1')

avg_disasters = np.zeros_like(disaster_data, dtype='float')
for i, year in enumerate(years):
    idx = year < trace.posterior.s.values
    avg_disasters[i] = ((trace.posterior.early_rate.values[idx].sum()
                        + trace.posterior.late_rate.values[~idx].sum())
                        / (len(trace.sample_stats.chain)
                           * len(trace.sample_stats.draw)))

# Highest Posterior Density (minimum width Bayesian credible interval)
# sp_hpd = pm.stats.hpd(trace.posterior.s)
sp_df = az.summary(trace.posterior.s)
ax.fill_betweenx(x1=sp_df['hdi_3%'], x2=sp_df['hdi_97%'],
                  y=[disaster_data.min(), disaster_data.max()],
                  alpha=0.5, color='C1')

ax.plot(years, avg_disasters, 'k--', lw=2)

plt.show()
# =============================================================================
# =============================================================================
