#!/usr/bin/env python3
# =============================================================================
#     File: orderlogistic_test.py
#  Created: 2024-01-19 11:30
#   Author: Bernie Roesler
#
"""
Example from:
<https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.OrderedLogistic.html>
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

# Generate data for a simple 1 dimensional example problem
n1_c = 300
n2_c = 300
n3_c = 300
cluster1 = np.random.randn(n1_c) + -1
cluster2 = np.random.randn(n2_c) + 0
cluster3 = np.random.randn(n3_c) + 2

x = np.concatenate((cluster1, cluster2, cluster3))
y = np.concatenate((1*np.ones(n1_c),
                    2*np.ones(n2_c),
                    3*np.ones(n3_c))) - 1

# Ordered logistic regression
with pm.Model() as model:
    cutpoints = pm.Normal("cutpoints", mu=[-1, 1], sigma=10, shape=2,
                          transform=pm.distributions.transforms.ordered)
    y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=x, observed=y)
    idata = pm.sample()

# Plot the results
plt.hist(cluster1, 30, alpha=0.5)
plt.hist(cluster2, 30, alpha=0.5)
plt.hist(cluster3, 30, alpha=0.5)
posterior = idata.posterior.stack(sample=("chain", "draw"))
plt.hist(posterior["cutpoints"][0], 80, alpha=0.2, color='k')
plt.hist(posterior["cutpoints"][1], 80, alpha=0.2, color='k')

# =============================================================================
# =============================================================================
