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

from scipy import stats


def metropolis(target, step=1, S=200, ax=None):
    r"""Sample from the distribution using the Metropolis algorithm.

    Parameters
    ----------
    target : :obj:`stats.*_frozen`
        The target distribution from which to sample. A frozen probability
        distribution from ``scipy.stats``.
    step : float
        The standard deviation of the jumping distribution. For this simple
        example, the jumping distribution is assumed to be:

        .. math:
            J_t(\theta^\ast | θ^{t-1}) \sim
                \mathcal{N}(\theta^\ast | \theta^{t-1}, \mathtt{step}^2 I).

    S : int, optional
        Number of samples.
    ax : plt.Axes
        Axes on which to plot the points.

    Returns
    -------
    samples : single item or (S,) ndarray
        The generated random samples.
    """
    rng = np.random.default_rng(seed=565656)
    I = np.eye(2)
    # I[0, 1] = I[1, 0] = -0.2  # TODO try correlated jumping distribution?

    θ_0 = (-1., 0.75)  # manual initial point
    if ax is not None:
        ax.scatter(*θ_0, marker='x', c='C3')

    samples = [θ_0]
    θ_tm1 = θ_0
    accepts = 0
    accept = True

    while len(samples) < S:
        # Define the jumping distribution centered about the last guess, and
        # sample a proposal value
        θ_p = rng.multivariate_normal(θ_tm1, step**2 * I)

        # Compute the ratio of the densities
        r = np.exp(np.log(target.pdf(θ_p)) - np.log(target.pdf(θ_tm1)))

        # Select the next sample with probability min(r, 1)
        if rng.random() < np.min((r, 1)):
            accept = True
            θ_t = θ_p
            accepts += 1
        else:
            accept = False
            θ_t = θ_tm1

        # Plot the proposed point as accepted or rejected
        if ax is not None:
            fc = 'k' if accept else 'none'
            ax.scatter(*θ_p, edgecolors='k', facecolors=fc, s=30)
            # Plot random walk path
            # ax.plot(*np.c_[θ_tm1, θ_t], lw=1, c='k')

        # Prepare for next step
        samples.append(θ_t)
        θ_tm1 = θ_t

    ax.set_title(f"step size = {step:.2f}, accept rate = {accepts/S:.2f}")
    return np.array(samples)


# Generate distribution with high correlation
ρ = -0.9
rv = stats.multivariate_normal(mean=[0, 0], cov=[[1, ρ], [ρ, 1]])

# Plot contours of the pdf on a uniform grid
xr = 1.5
x0, x1 = np.mgrid[-xr:xr:0.01, -xr:xr:0.01]
pos = np.dstack((x0, x1))

# Figure 9.3
fig, axs = plt.subplots(num=1, ncols=2, clear=True, sharey=True)

# (a) step size = 0.1 -> accept rate = 0.62
# (b) step size = 0.25 -> accept rate = 0.34
samples_a = metropolis(rv, step=0.10, S=50, ax=axs[0])
samples_b = metropolis(rv, step=0.25, S=50, ax=axs[1])

for ax in axs:
    ax.contour(x0, x1, rv.pdf(pos), zorder=0)
    ax.set(xlabel=r'$x_0$',
           ylabel=r'$x_1$',
           aspect='equal')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
