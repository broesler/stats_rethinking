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

from scipy import stats


def sample_dimensions(D=10, T=1000, N_bins=50):
    """Sample a D-dimensional multivariate normal.

    Parameters
    ----------
    D : int, optional
        Number of dimensions.
    T : int, optional
        Number of samples.
    N_bins : int, optional
        Number of bins to create the histogram.

    Returns
    -------
    dens : (N_bins,) ndarray of float
        The probability density of the data.
    bin_centers : (N_bins,) ndarray of float
        The locations of the bin centers.
    """
    Y = stats.multivariate_normal(np.zeros(D), np.eye(D)).rvs(T)  # (T, D)
    if D > 1:
        Rd = np.sqrt(np.sum(Y**2, axis=1))  # (T,)
    else:
        Rd = np.sqrt(Y**2)

    dens, bin_edges = np.histogram(Rd, bins=50, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return dens, bin_centers


fig = plt.figure(1, clear=True)
ax = fig.add_subplot()

for D in [1, 10, 100, 1000]:
    dens, bin_centers = sample_dimensions(D)
    ax.plot(bin_centers, dens, c='k')
    idx = np.argmax(dens)
    ax.text(s=f"{D}", x=bin_centers[idx], y=dens[idx] + 0.02, ha='center')

ax.set(xlabel='Radial distance from mode',
       ylabel='Density')

# =============================================================================
# =============================================================================
