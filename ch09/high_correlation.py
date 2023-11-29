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
import xarray as xr

from scipy import stats

rng = np.random.default_rng(seed=5656)


def metropolis(target, S=200, init=None, step=1):
    r"""Sample from the distribution using the Metropolis algorithm.

    Parameters
    ----------
    target : :obj:`stats.*_frozen`
        The target distribution from which to sample. A frozen probability
        distribution from ``scipy.stats``.
    S : int, optional
        Number of samples to keep.
    init : (N,) tuple
        The initial value(s) from which to start the Markov chain. Must be the
        same number of dimensions as the target distribution.
    step : float
        The standard deviation of the jumping distribution. For this simple
        example, the jumping distribution is assumed to be:

        .. math:
            J_t(\theta^\ast | θ^{t-1}) \sim
                \mathcal{N}(\theta^\ast | \theta^{t-1}, \mathtt{step}^2 I).

    Returns
    -------
    samples : single item or (S,) ndarray
        The generated random samples.
    rejects : single item or (M,) ndarray
        The rejected proposed sample points.
    """
    I = np.eye(target.dim)

    if init is None:
        init = np.zeros(target.dim)
    else:
        assert len(init) == target.dim

    θ_tm1 = init
    samples = [θ_tm1]
    rejects = []

    while len(samples) < S:
        # Define the jumping distribution centered about the last guess, and
        # sample a proposal value
        θ_p = rng.multivariate_normal(θ_tm1, step**2 * I)

        # Compute the ratio of the densities
        r = np.exp(np.log(target.pdf(θ_p)) - np.log(target.pdf(θ_tm1)))

        # Select the next sample with probability min(r, 1)
        if rng.random() < np.min((r, 1)):
            θ_tm1 = θ_p
            samples.append(θ_p)
        else:
            rejects.append(θ_p)

    return (np.array(samples), np.array(rejects))


# Replicate Gelman Figure 11.1 with 5 starting locations of centered Gaussian.
norm_2d = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2))

inits = np.array([
    [2.5, 2.5],
    [-2.5, 2.5],
    [0., 0.],
    [-2.5, -2.5],
    [2.5, -2.5],
])

S = 2000
samp = xr.DataArray(
    dims=('chain', 'draw', 'x_dim_0'),
    coords=dict(
        chain=np.arange(inits.shape[0]),
        draw=np.arange(S),
        x_dim_0=np.arange(2),
    ),
)

fig, axs = plt.subplots(num=1, ncols=3, sharex=True, sharey=True, clear=True)
fig.set_size_inches((12, 4), forward=True)

for chain, init in enumerate(inits):
    samples, rejects = metropolis(norm_2d, S=2000, init=init, step=0.2)
    samp.loc[dict(chain=chain)] = samples

    # Line plot of N iterations showing random walk effect
    for ax, N in zip(axs[:2], [50, 1000]):
        ax.scatter(*init, c='k', marker='s')
        ax.plot(*samp.sel(dict(chain=chain, draw=range(N))).T, c='k', lw=1)

    # Scatter plot of the iterates of the second halves of the sequences
    # TODO jitter points so steps where random walks stood still are visible.
    axs[2].scatter(*samp.sel(dict(chain=chain, draw=range(1000, 2000))).T,
                   c='k', s=1, alpha=0.5)

# Format plots
fig.suptitle('Gelman BDA3, Figure 11.1', fontweight='bold')
axs[0].set(xlim=(-4, 4), ylim=(-4, 4))
axs[0].set_title('First 50 iterations')
axs[1].set_title('First 1000 iterations')
axs[2].set_title('Last 1000 draws')
for ax in axs:
    ax.set(aspect='equal')


# -----------------------------------------------------------------------------
#         Figure 9.3
# -----------------------------------------------------------------------------
# Generate distribution with high correlation
ρ = -0.9
rv = stats.multivariate_normal(mean=[0, 0], cov=[[1, ρ], [ρ, 1]])

# Plot contours of the pdf on a uniform grid
xm = 2.5
x0, x1 = np.mgrid[-xm:xm:0.01, -xm:xm:0.01]
pos = np.dstack((x0, x1))

fig, axs = plt.subplots(num=2, ncols=2, sharey=True, clear=True)
fig.set_size_inches((10, 5), forward=True)

# (a) step size = 0.1 -> accept rate = 0.62
# (b) step size = 0.25 -> accept rate = 0.34
S = 50

for ax, step in zip(axs, [0.1, 0.25]):
    # Extract the samples
    init = (-1., 0.75)
    samples, rejects = metropolis(rv, S, init=init, step=step)

    # Plot the contours of the target pdf
    ax.contour(x0, x1, rv.pdf(pos), zorder=0)

    # Plot the accepted/rejected points
    ax.scatter(samples[:, 0], samples[:, 1], c='k', s=20)
    ax.scatter(rejects[:, 0], rejects[:, 1], ec='k', fc='none', s=20)

    # Plot the initial point
    ax.scatter(*init, marker='x', c='C3')

    ax.set(
        title=(f"step size = {step:.2f},"
               f"accept rate = {S/(len(rejects) + S):.2f}"),
        xlabel=r'$x_0$',
        ylabel=r'$x_1$',
        aspect='equal'
    )

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
