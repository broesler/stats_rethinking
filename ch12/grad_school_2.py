#!/usr/bin/env python3
# =============================================================================
#     File: grad_school_2.py
#  Created: 2024-01-16 19:45
#   Author: Bernie Roesler
#
"""
§12.1 Over-Dispersed models.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from cycler import cycler
from pathlib import Path
from scipy import stats

import stats_rethinking as sts


# ----------------------------------------------------------------------------- 
#         Test plot of beta distribution
# -----------------------------------------------------------------------------
def beta(pbar, theta):
    # Convert to standard parameterization
    a = pbar * θ
    b = (1 - pbar) * θ 
    return stats.beta(a, b)

x = np.linspace(0, 1, 200)
pbars = np.arange(0.1, 1, 0.1)
θs = [0.1, 1, 2, 3, 5, 10, 30, 50, 100]

fig, axs = plt.subplots(num=1, ncols=2, sharex=True, sharey=True, clear=True)

ax = axs[0]
ax.set_prop_cycle(
    cycler('color', plt.cm.viridis(np.linspace(0, 0.9, len(pbars))))
)

# Vary pbar
θ = 10
for pbar in pbars:
    ax.plot(x, beta(pbar, θ).pdf(x),
            label=(f"{pbar = :g}, "
                   rf"($\alpha$ = {pbar*θ:.1f}, $\beta$ = {(1 - pbar)*θ:.1f})")
            )

ax.legend()
ax.set(xlabel='x',
       ylabel='density',
       xlim=(0, 1))
ax.spines[['top', 'right']].set_visible(False)

# Vary θ
ax = axs[1]
ax.set_prop_cycle(
    cycler('color', plt.cm.viridis(np.linspace(0, 0.9, len(θs))))
)

pbar = 0.3
ax.axvline(pbar, c='k', ls='--', lw=1, alpha=0.3)

for θ in θs:
    ax.plot(x, beta(pbar, θ).pdf(x),
            label=(f"{θ = :3g}, "
                   rf"($\alpha$ = {pbar*θ:.1f}, $\beta$ = {(1 - pbar)*θ:.1f})")
            )

ax.legend()
ax.set(xlabel='x')
ax.spines[['top', 'right']].set_visible(False)

# =============================================================================
# =============================================================================
