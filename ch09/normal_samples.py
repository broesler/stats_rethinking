#!/usr/bin/env python3
# =============================================================================
#     File: hmc.py
#  Created: 2023-11-30 10:55
#   Author: Bernie Roesler
#
"""
Example script for Hamiltonian Monte Carlo method.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats

# Assuem we only have a uniform random variable
U = stats.uniform().rvs(size=1000)

# Take the invers CDF to get a normally-distributed rv
x = stats.norm.ppf(U)

# Compute the exact pdf values for comparison
xe = np.linspace(stats.norm.ppf(0.001), stats.norm.ppf(0.999), 1000)

fig, ax = plt.subplots(num=1, clear=True)
sns.histplot(x, stat='density', kde=True, ax=ax)
ax.plot(xe, stats.norm.pdf(xe), 'k')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
