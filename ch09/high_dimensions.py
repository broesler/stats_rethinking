#!/usr/bin/env python3
# =============================================================================
#     File: high_dimensions.py
#  Created: 2023-11-22 18:29
#   Author: Bernie Roesler
#
"""
Plot Figure 9.4 with simulations of distance from mode.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats


def sample_dimensions(D=10, T=1000, N_bins=50, sort=True):
    """Sample a D-dimensional multivariate normal.

    Parameters
    ----------
    D : int, optional
        Number of dimensions.
    T : int, optional
        Number of samples.
    N_bins : int, optional
        Number of bins to create the histogram.
    sort : bool, optional
        If True, sort the output array.

    Returns
    -------
    Rd : (T,) ndarray of float
        The sampled values.
    """
    Y = stats.multivariate_normal(np.zeros(D), np.eye(D)).rvs(T)  # (T, D)
    Rd = np.sqrt(np.sum(Y**2, axis=1)) if D > 1 else np.abs(Y)

    if sort:
        Rd = np.sort(Rd)

    return Rd


fig = plt.figure(1, clear=True)
fig.set_size_inches((10, 3), forward=True)
ax = fig.add_subplot()

for D in [1, 10, 100, 1000]:
    Rd = sample_dimensions(D)
    kde = stats.gaussian_kde(Rd, bw_method=0.1)
    dens = kde.pdf(Rd)
    ax.plot(Rd, dens, c='k')
    idx = np.argmax(dens)
    ax.text(s=f"{D}", x=Rd[idx], y=dens[idx] + 0.02, ha='center')

ax.set(xlabel='Radial distance from mode',
       ylabel='Density',
       ylim=(0, 1))
ax.spines[['right', 'top']].set_visible(False)

# =============================================================================
# =============================================================================
