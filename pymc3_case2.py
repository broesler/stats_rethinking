#!/usr/bin/env python3
#==============================================================================
#     File: pymc3_case2.py
#  Created: 2019-06-20 23:24
#   Author: Bernie Roesler
#
"""
  Description: Case study of coal mine accident data.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

plt.style.use('seaborn-darkgrid')
np.random.seed(123)  # initialize random number generator

disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                           3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                           2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
                           1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                           3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = np.arange(1851, 1962)

fig, ax = plt.subplots(num=1, clear=True)
ax.plot(years, disaster_data, 'o', markersize=8)
ax.set(ylabel='Disaster count',
       xlabel='Year')

plt.show()

with pm.Model() as disaster_model:
    # Switch point when disaster rate reduced
    s = pm.DiscreteUniform('s', lower=years.min(), upper=years.max(), testval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(years <= s, early_rate, late_rate)

    disasters = pm.Poisson('disasters', rate, observed=disaster_data)

with disaster_model:
    trace = pm.sample(10000)

pm.traceplot(trace)
#==============================================================================
#==============================================================================
