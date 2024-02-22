#!/usr/bin/env python3
# =============================================================================
#     File: gaussian_ellipses.py
#  Created: 2024-02-21 13:29
#   Author: Bernie Roesler
#
"""
Toy script to plot Gaussian confidence ellipses.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import transforms
from matplotlib.patches import Circle, Ellipse
from scipy import linalg, stats

# TODO refactor `level` into `q`?
def confidence_ellipse(mean, cov, ax=None,
                       level=0.95, facecolor='none', **kwargs):
    """Plot an ellipse showing the confidence region of a 2D Gaussian.

    Parameters
    ----------
    mean : (2,) array_like
        The center of the ellipse.
    cov : (2, 2) array_like
        The covariance matrix.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    level : float, optional
        The level of the confidence interval to plot. Default is 0.95.
    facecolor : str, optional
        The color with which to fill the ellipse. Default is 'none'.
    **kwargs : dict
        Additional arguments to pass to the Ellipse call.

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The ellipse object.

    References
    ----------
    [0]: <https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html>
    """
    assert mean.shape == (2,), "Mean must be length 2"
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2"
    ax = ax or plt.gca()

    # The scaling parameter is chi-squared with N dof for N-D data
    s = np.sqrt(stats.chi2.ppf(level, df=2))
    # s = -stats.norm.ppf((1 - level) / 2)  # WRONG

    # Σ = [[σ_x^2, σ_x σ_y ρ],
    #      [σ_x σ_y ρ, σ_y^2]]
    # => pearson == ρ
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Use a special case to obtain the eigenvalues of this 2D dataset.
    r_x = np.sqrt(1 + pearson)
    r_y = np.sqrt(1 - pearson)
    scale = np.sqrt(np.diag(cov)) * s

    transform = (
        transforms.Affine2D()
        .rotate_deg(45)  # == arctan2(v[1], v[0]), v == eigenvector of cov
        .scale(*scale)
        .translate(*mean)
    )

    ellipse = Ellipse((0, 0),
                      width=2*r_x,
                      height=2*r_y,
                      facecolor=facecolor,
                      transform=transform + ax.transData,
                      **kwargs)

    return ax.add_patch(ellipse)


def confidence_circle(mean, cov, ax=None,
                      level=0.95, facecolor='none', **kwargs):
    """Plot a confidence ellipse for a 2D Gaussian.

    Parameters
    ----------
    mean : (N,) array_like
        The center of the ellipse.
    cov : (N, N) array_like
        The covariance matrix. Must be symmetric positive semi-definite.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    level : float, optional
        The level of the confidence interval to plot. Default is 0.95.
    facecolor : str, optional
        The color with which to fill the ellipse. Default is 'none'.
    **kwargs : dict
        Additional arguments to pass to the Ellipse call.

    Returns
    -------
    circle : matplotlib.patches.Circle
        The circle object.

    References
    ----------
    [0]: <https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712?permalink_comment_id=3465086istcomment-3465086>
    """
    assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"
    ax = ax or plt.gca()
    kwargs.update(dict(facecolor=facecolor))
    # The scaling parameter is chi-squared with N dof for N-D data
    r = np.sqrt(stats.chi2.ppf(level, df=2))
    # Transform matrix is 3x3 to include translation
    T = np.eye(3)
    T[:2, :2] = r * linalg.sqrtm(cov)  # scaling and rotation
    transform = transforms.Affine2D(matrix=T).translate(*mean)
    circle = Circle(
        xy=(0, 0),
        radius=1,
        transform=transform + ax.transData,
        **kwargs
    )
    return ax.add_patch(circle)


# -----------------------------------------------------------------------------
#         Define constants
# -----------------------------------------------------------------------------
np.random.seed(565656)
σ_x = 0.9
σ_y = 0.8
ρ = 0.7

μ = np.r_[np.pi, -1]
cov_xy = σ_x * σ_y * ρ
Σ = np.array([[σ_x**2, cov_xy],
              [cov_xy, σ_y**2]])

rv = stats.multivariate_normal(μ, Σ)

N = 1000
samples = rv.rvs(N)
x, y = samples.T

# Grid of Gaussian values to plot contours
xg, yg = np.mgrid[x.min():x.max():0.01, y.min():y.max():0.01]
zg = rv.pdf(np.dstack((xg, yg)))
zg = (zg.max() - zg) / zg.max()  # invert and normalize

# qs = np.r_[0.5]  # breaks for contour plot
qs = np.r_[0.6827, 0.9545]  # == [1, 2] std deviations for 1-D Gaussian
# n_stds = -stats.norm.ppf((1 - qs) / 2)  # [1.0000, 2.0000]

# Color in/outside of ellipse
r = np.sqrt(stats.chi2.ppf(qs[0], df=2))
# Scale the samples to the unit circle for comparison
u, v = linalg.inv(linalg.sqrtm(Σ)) @ (samples - μ).T
inside = u**2 + v**2 <= r**2
outside = ~inside

# -----------------------------------------------------------------------------
#         Plot the slopes and intercepts
# -----------------------------------------------------------------------------
# Create a custom colormap for the contours
colors = plt.get_cmap('Blues')(np.linspace(0.75, 1.0, len(qs)))
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('Blues', colors)
fmt = {q: f"{-stats.norm.ppf((1 - q) / 2):.0g}σ" for q in qs}

fig, ax = plt.subplots(num=1, clear=True)

# Plot the numerical contours
cs = ax.contour(xg, yg, zg, cmap=cmap, levels=qs)
ax.clabel(cs, fmt=fmt)

for level in qs:
    E = confidence_ellipse(μ, Σ, ax=ax, level=level,
                           ec='k', ls='-.', lw=3, alpha=0.4, zorder=3)
    C = confidence_circle(μ, Σ, ax=ax, level=level,
                          ec='C3', ls='--', lw=3, zorder=2)

# ax.scatter(*samples, c='C0', alpha=0.2)
ax.scatter(*samples[inside].T, c='C0', alpha=0.2)
ax.scatter(*samples[outside].T, c='C3', alpha=0.2)

ax.set(title=f"confidence = {qs[0]:.3f}, empirical = {sum(inside)/N:.3f}",
       xlabel='x',
       ylabel='y',
       aspect='equal')

# TODO investigate: np.allclose(stats.chi2.ppf(0.95, df=2), -2*np.log(0.05))
# ----------------------------------------------------------------------------- 
#         Plot error in `n_std` calculation
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=2, clear=True)
stds = np.linspace(0.1, 8.5)
qs = 1 - 2*stats.norm.cdf(-stds)
rs = np.sqrt(stats.chi2.ppf(qs, df=2))
ax.plot(stds, (rs - stds) / stds, '.-')
ax.axhline(0, ls='--', c='gray', lw=1)
ax.set_xticks(np.arange(9))
ax.set(xlabel='σ', ylabel='% difference from χ²')

# =============================================================================
# =============================================================================
