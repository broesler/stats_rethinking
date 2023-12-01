#!/usr/bin/env python3
# =============================================================================
#     File: hamiltonian.py
#  Created: 2023-11-30 11:53
#   Author: Bernie Roesler
#
"""
Example Hamiltonian Monte Carlo method.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from scipy import stats


# (R code 9.3) U needs to return neg-log-probability
def myU4(q, x, y, m_x=0, s_x=0.5, m_y=0, s_y=0.5):
    r"""Compute the negative log probability of a 2D gaussian.

    The model is as follows:

    .. math::
        x_i ~ \mathcal{N}(\mu_x, 1)
        y_i ~ \mathcal{N}(\mu_y, 1)
        \mu_x ~ \mathcal{N}(0, 0.5)
        \mu_y ~ \mathcal{N}(0, 0.5).

    Parameters
    ----------
    q : (2,) array_like
        Vector of parameter values [μ_x, μ_y].
    x, y : (N,) array_like
        The data.
    m_[xy] : float
        The means of the underlying normal distributions.
    s_[xy] : float
        The standard deviations of the underlying normal distributions.

    Returns
    -------
    result : (2,) ndarray
        The negative log probabilities of the parameters ``q``.
    """
    μ_x, μ_y = q
    U = (
          np.sum(np.log(stats.norm(μ_x, 1).pdf(x)))
        + np.sum(np.log(stats.norm(μ_y, 1).pdf(y)))
        + stats.norm(m_x, s_x).pdf(μ_x)
        + stats.norm(m_y, s_y).pdf(μ_y)
    )
    return -U


# (R code 9.4) gradient function
# need vector of partial derivatives of U wrt vector q
def myU_grad4(q, x, y, m_x=0, s_x=0.5, m_y=0, s_y=0.5):
    r"""Compute the gradient of the 2D Gaussian.

    Parameters
    ----------
    q : (2,) array_like
        Vector of parameter values [μ_x, μ_y].
    x, y : (N,) array_like
        The data.
    m_[xy] : float
        The means of the underlying normal distributions.
    s_[xy] : float
        The standard deviations of the underlying normal distributions.

    Returns
    -------
    result : (2,) ndarray
        The gradient of the 2D Gaussian at the location ``q``.
    """
    μ_x, μ_y = q
    G_x = np.sum(x - μ_x) + (m_x - μ_x) / s_x**2
    G_y = np.sum(y - μ_y) + (m_y - μ_y) / s_y**2
    return np.r_[-G_x, -G_y]  # negative because energy is neg-log-prob


# TODO
# * clean up this code, many useless lines (i.e. p = -p)
# * make a wrapper loop to generate S samples in C chains?
def hamiltonian_sample(q0, x, y, U, grad_U, step=0.1, L=10, **kwargs):
    """Compute a Hamiltonian Monte Carlo simulation to generate one sample.

    Parameters
    ----------
    q0 : (N,) array_like
        The vector of parameters from which to generate a sample.
    U, grad_U : callable
        The target distribution function and its gradient.
    step : float, optional
        The step size.
    L : int, optional
        The number of leapfrog steps to take along the trajectory.
    **kwargs : dict_like, optional
        Additional function arguments passed to ``U`` and ``grad_U``.

    Returns
    -------
    result : dict
        Dictionary with fields:
            q : float
                The sample value.
            traj, ptraj : (L, N) ndarrays
                The position (parameter values) and momentum arrays for the
                intermediate leapfrog steps.
            accept : bool
                True if the sample is accepted.
            dH : (N,) ndarray
                The difference in the proposed and current sums of position and
                momentum.
    """
    # Initial position and momentum
    q = q0
    p = rng.normal(size=q.shape)
    p0 = p

    # Make a half-step for momentum at the beginning
    p -= step * grad_U(q, x, y, **kwargs) / 2

    # Initialize the trajectory
    qt = np.empty((L+1, len(q)))
    pt = np.empty((L+1, len(q)))
    qt[:] = np.nan
    pt[:] = np.nan
    qt[0] = q
    pt[0] = p

    # Alternate full steps for position and momentum
    for i in range(1, L+1):
        q += step * p  # full step for position
        if i != L:
            p -= step * grad_U(q, x, y, **kwargs)
            pt[i] = p
        qt[i] = q

    # Make a half step for momentum at the end
    p -= step * grad_U(q, x, y, **kwargs) / 2
    pt[L] = p

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p

    # Evaluate potential and kinetic energies at start and end of trajectory
    U0 = U(q0, x, y, **kwargs)
    K0 = np.sum(p0**2) / 2
    Up = U(q, x, y, **kwargs)
    Kp = np.sum(p**2) / 2
    # Compute the log Hamiltonians
    H0 = U0 + K0
    Hp = Up + Kp

    # Accept or reject the state at the end
    accept = rng.random() < np.exp(H0 - Hp)
    new_q = q if accept else q0

    # TODO convert this dictionary into a class for easier documentation/use
    return dict(
        q=new_q,
        qt=qt,
        pt=pt,
        accept=accept,
        dH=Hp - H0,
    )


# -----------------------------------------------------------------------------
#         Test Data
# -----------------------------------------------------------------------------
rng = np.random.default_rng(seed=5656)

# Generate data
x = rng.normal(size=50)
y = rng.normal(size=50)
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

q0 = np.r_[-0.1, 0.2]

# TODO
# * make Figure 9.6:
#   (a) L = 11, 2D standard normal
#   (b) L = 28, 2D standard normal
#   (c) L = ?, ρ = -0.9
#   (d) L = ?, ρ = -0.9, 50 trajectories
# * turn plotting into a function over which to loop

# NOTE
# Figure 9.6
#   (a) N = 4,  L = 11, step = 0.03, q0 = [-0.1, 0.2]
#   (b) N = 2,  L = 28, step = 0.03, q0 = [-0.1, 0.2]

N = 4  # number of samples
L = 11

# Plot contours of the pdf on a uniform grid
xm = 0.5
x0, x1 = np.mgrid[-xm:xm:0.01, -xm:xm:0.01]
pos = np.dstack((x0, x1))

# TODO how to fix U, grad_U to use this correlation??
ρ = -0.9
target = stats.multivariate_normal(mean=[0, 0], cov=np.c_[[1, ρ], [ρ, 1]])

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()

# Plot the contours of the target pdf
ax.contour(x0, x1, target.pdf(pos), levels=6, zorder=0)

# Plot the starting point
ax.scatter(*q0, c='r', marker='x')

for i in range(N):
    # Get a single sample
    Q = hamiltonian_sample(q0, x, y, myU4, myU_grad4, step=0.03, L=L)

    if N < 11:
        # Plot the trajectory with varying linewidth
        KE = np.sum(Q['pt']**2 / 2, axis=1)
        lws = 1 + KE
        pts = Q['qt'][:, np.newaxis, :]  # (L, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (L-1, 2, 2)
        lc = LineCollection(segs, linewidths=lws, color='k', alpha=0.4)
        ax.add_collection(lc)
        # Plot the trajectory points
        ax.scatter(*Q['qt'].T, c='k', s=2, alpha=0.4)

        # Label the sample points
        ax.annotate(
            text=f"{i}",
            xy=Q['qt'][-1],
            xytext=(5, 5),
            textcoords='offset points',
        )

    # Plot the sample point (last leapfrog iteration)
    ax.scatter(
        *Q['qt'][-1],
        ec='k',
        fc='r' if Q['dH'] > 0.1 else ('k' if Q['accept'] else 'none'),
    )

ax.set(
    xlabel=r'$\mu_x$',
    ylabel=r'$\mu_y$',
    xlim=(-xm, xm),
    ylim=(-xm, xm),
    aspect='equal',
)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
