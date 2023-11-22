#!/usr/bin/env python3
# =============================================================================
#     File: high_correlation.py
#  Created: 2023-11-21 16:12
#   Author: Bernie Roesler
#
"""
§9.2 code with highly correlated variables.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

# Generate distribution with high correlation
ρ = -0.9
rv = stats.multivariate_normal(cov=[[1, ρ], [ρ, 1]])

# TODO sample the distribution using a Metropolis chain
# (a) step size = 0.1 -> accept rate = 0.62
# (b) step size = 0.25 -> accept rate = 0.34

# Steps:
# 1. "Random" initial guess (plot with 'x')
# 2. Accept or reject (plot with open or closed circle)
# 3. Store accepted points as "the sample"
# 4. Compute the acceptance rate.

# Plot contours of the pdf on a uniform grid
x0, x1 = np.mgrid[-1:1:0.01, -1:1:0.01]
pos = np.dstack((x0, x1))

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.contour(x0, x1, rv.pdf(pos))
ax.set(xlabel=r'$x_0$',
       ylabel=r'$x_1$',
       aspect='equal')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
