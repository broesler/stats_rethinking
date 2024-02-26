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
import pandas as pd
import warnings

from matplotlib import transforms
from matplotlib.patches import Circle, Ellipse
from pathlib import Path
from scipy import linalg, stats

path = Path('/Users/bernardroesler/src/web_dev/broesler.github.io/assets/images/conf_ellipse')


def confidence_ellipse(mean, cov, ax=None, n_std=None, level=None,
                       facecolor='none', **kwargs):
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

    if n_std is not None and level is not None:
        raise ValueError("Must specify only one of `n_std` or `level`.")

    if n_std is not None:
        level = stats.chi2.cdf(n_std**2, df=2)

    # The scaling parameter is chi-squared with N dof for N-D data
    s = np.sqrt(stats.chi2.ppf(level, df=2))

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


def confidence_circle(mean, cov, ax=None, n_std=None, level=None,
                      facecolor='none', **kwargs):
    """Plot a confidence ellipse for a 2D Gaussian.

    Parameters
    ----------
    mean : (2,) array_like
        The center of the ellipse.
    cov : (2, 2) array_like
        The covariance matrix. Must be symmetric positive semi-definite.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    n_std : float, optional
        The number of standard deviations from the mean to plot the ellipse, as
        computed by the Mahalanobis distance.
        One of `n_std` or `level` must be specified.
    level : float, optional
        The level of the confidence interval to plot.
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
    assert mean.shape == (2,), "Mean must be length 2"
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2"
    ax = ax or plt.gca()
    kwargs.update(dict(facecolor=facecolor))

    if n_std is None and level is None:
        raise ValueError("Must specify either `n_std` or `level`.")

    if n_std is not None:
        level = 1 - np.exp(-n_std**2 / 2)  # == stats.chi2.cdf(n_std**2, df=2)
        if level is not None:
            warnings.warn("Ignoring `level` parameter; using `n_std` instead.")

    # The scaling parameter is chi-squared with N dof for N-D data
    # == np.sqrt(stats.chi2.ppf(level, df=2))
    r = np.sqrt(-2 * np.log(1 - level))

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


# TODO add labels to the vectors
def plot_svd_vecs(A, ax=None, **kwargs):
    """Plot the vectors of the SVD of a 2x2 matrix.

    Parameters
    ----------
    A : (2, 2) array_like
        The matrix to decompose.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    **kwargs : dict
        Additional arguments to pass to the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object.
    """
    assert A.shape == (2, 2), "Matrix must be 2x2"
    ax = ax or plt.gca()
    # Take the SVD of A
    U, s, Vt = linalg.svd(A)
    # Plot the vectors of Vt on the unit circle
    ax.quiver(0, 0, *Vt[0], angles='xy', scale_units='xy', scale=1, color='C0')
    ax.quiver(0, 0, *Vt[1], angles='xy', scale_units='xy', scale=1, color='C0')
    ax.add_patch(Circle((0, 0), 1, fill=False, color='k', ls='--', lw=1, zorder=1))
    # Plot the scaled vectors U @ S on the corresponding ellipse
    ax.quiver(0, 0, *(U @ np.sqrt(np.diag(s)))[:, 0], angles='xy', scale_units='xy', scale=1, color='C3') 
    ax.quiver(0, 0, *(U @ np.sqrt(np.diag(s)))[:, 1], angles='xy', scale_units='xy', scale=1, color='C3')
    ax.add_patch(
        Ellipse(
            (0, 0),
            width=2*np.sqrt(s[0]),
            height=2*np.sqrt(s[1]),
            angle=np.degrees(np.arctan2(U[1, 0], U[0, 0])),
            fill=False, color='k', ls='--', lw=1, zorder=1,
        )
    )
    return ax


# -----------------------------------------------------------------------------
#         Define constants
# -----------------------------------------------------------------------------
np.random.seed(565656)
σ_x = 1.0
σ_y = 0.8
ρ = 0.7

μ = np.r_[3, -1]
# μ = np.r_[0, 0]
cov_xy = σ_x * σ_y * ρ
Σ = np.array([[σ_x**2, cov_xy],
              [cov_xy, σ_y**2]])

rv = stats.multivariate_normal(μ, Σ)

N = 1000
samples = rv.rvs(N)
x, y = samples.T

# Estimate the population parameters from the sample
μ_hat = samples.mean(axis=0)
Σ_hat = np.cov(samples.T, ddof=1)

# TODO separate code for initial "1σ" ellipses
qs = np.r_[0.683]
n_std = np.sqrt(stats.chi2.ppf(qs, df=2))

# Grid of Gaussian values to plot contours
x_min = min(x.min() - 0.2, μ[0] - (n_std + 1)*σ_x)
x_max = max(x.min() + 0.2, μ[0] + (n_std + 1)*σ_x)
y_min = min(y.min() - 0.2, μ[1] - (n_std + 1)*σ_y)
y_max = max(y.min() + 0.2, μ[1] + (n_std + 1)*σ_y)

xg, yg = np.mgrid[x_min:x_max:0.01, y_min:y_max:0.01]
# zg = rv.pdf(np.dstack((xg, yg)))
zg = stats.multivariate_normal(μ_hat, Σ_hat).pdf(np.dstack((xg, yg)))
zg = (zg.max() - zg) / zg.max()  # invert and normalize

# TODO use matplotlib.patches.Patch.contains_point instead
# Color in/outside of ellipse
r_sq = stats.chi2.ppf(qs[0], df=2)

# r = 1  # what level does this correspond to?
# => level = stats.chi2(2).cdf(r**2) == 0.3934693402873665

# Scale the samples to the unit circle for comparison
# u, v = linalg.inv(linalg.sqrtm(Σ)) @ (samples - μ).T
u, v = linalg.inv(linalg.sqrtm(Σ_hat)) @ (samples - μ_hat).T
inside = (u**2 + v**2) <= r_sq
outside = ~inside

# -----------------------------------------------------------------------------
#         Plot the slopes and intercepts
# -----------------------------------------------------------------------------
# Create a custom colormap for the contours
cm = 'Blues_r'
range_ = np.linspace(0.0, 0.5, len(qs))
colors = plt.get_cmap(cm)(range_)

if len(qs) > 1:
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(cm, colors)
else:
    cmap = cm

# fmt = {q: f"{-stats.norm.ppf((1 - q) / 2):.2g}σ" for q in qs}  # FIXME WRONG
fmt = {q: f"{np.sqrt(stats.chi2.ppf(q, df=2)):.3g}σ" for q in qs}

fig, ax = plt.subplots(num=1, clear=True)

# Plot the numerical contours
cs = ax.contour(xg, yg, zg, levels=qs, cmap=cmap)
ax.clabel(cs, fmt=fmt)

for i, level in enumerate(qs):
    # n_std = -stats.norm.ppf((1 - level)/2)
    E = confidence_ellipse(μ_hat, Σ_hat, ax=ax, n_std=n_std, #level=level,
                           ec=colors[i], ls='-.', lw=2, alpha=0.4, zorder=3)
    # E = confidence_ellipse(μ, Σ, ax=ax, n_std=n_std, #level=level,
    #                        ec=colors[i], ls='-.', lw=2, alpha=0.4, zorder=3)
    # C = confidence_circle(μ, Σ, ax=ax, level=level,
    #                       ec=colors[i], ls='--', lw=2, zorder=2)

# ax.scatter(*samples, c='C0', alpha=0.2)
pts_alpha = 0.2 if N > 100 else 0.6
ax.scatter(*samples[inside].T, c='C0', alpha=pts_alpha)
ax.scatter(*samples[outside].T, c='C3', alpha=pts_alpha)

ax.set(#title=f"confidence = {qs[0]:.3f}, empirical = {sum(inside)/N:.3f}",
       xlabel='x',
       ylabel='y',
       aspect='equal')
ax.spines[['top', 'right']].set_visible(False)

# if N == 20:
#     fig.savefig(path / Path('./initial_problem.pdf'), transparent=False)

# qs = np.r_[0.997]
# n_std = 3
# fig.savefig(path / Path('./three_sigma_wrong.pdf'), transparent=False)

# TODO
#  * investigate: np.allclose(stats.chi2.ppf(0.95, df=2), -2*np.log(0.05))
#  * linalg.eig(Σ) == linalg.svd(Σ) since Σ is symmetric
#  * play with eig/SVD to understand the rotation and scaling
# NOTE
# T[:2, :2] = r * linalg.cholesky(cov).T  # scaling and rotation works too

U, s, Vt = linalg.svd(Σ)
D = stats.multivariate_normal(cov=np.eye(2)).rvs(N).T

fig, ax = plt.subplots(num=3, clear=True)
ax.scatter(*D, c='gray', alpha=0.2, label=r"$D \sim \mathcal{N}(0, I)$")
ax.scatter(*(Vt @ D), c='C0', alpha=0.2, label=r"$V^T D$")
ax.scatter(*(np.sqrt(np.diag(s)) @ Vt @ D), c='C3', alpha=0.2, label=r"$\sqrt{S} V^T D$")
ax.scatter(*(U @ np.sqrt(np.diag(s)) @ Vt @ D), c='C2', alpha=0.2,
           label=r"$U \sqrt{S} V^T D = \Sigma D$")
ax.legend()
ax.set(title='Geometric Transformation via SVD',
       xlabel='x',
       ylabel='y',
       aspect='equal')

# -----------------------------------------------------------------------------
#         Define the probability table
# -----------------------------------------------------------------------------
ks = np.arange(1, 11)
sigs = np.arange(1, 7)
q_arr = (
    np.array([stats.chi2.cdf(z**2, df=k) for k in ks for z in sigs])
    .reshape(len(ks), len(sigs))
)
df = pd.DataFrame(q_arr, index=ks, columns=sigs)
df.index.name = 'dimensions'
df.columns.name = 'n_std'

# print(df.to_markdown(floatfmt='.4f'))


# Plot SVD vectors
A = np.array([[1, 2], 
              [0, 2]])

fig, ax = plt.subplots(num=4, clear=True)
plot_svd_vecs(A, ax=ax)

ax.set(title='SVD of A',
       xlabel='x',
       ylabel='y',
       aspect='equal')

# =============================================================================
# =============================================================================
