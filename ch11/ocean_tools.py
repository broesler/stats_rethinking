#!/usr/bin/env python3
# =============================================================================
#     File: ocean_tools.py
#  Created: 2023-12-13 20:25
#   Author: Bernie Roesler
#
r"""
§11.2 Poisson models.

Fit the model:

.. math::
    T \sim \mathrm{Poisson}(\lambda)
    \log \lambda = \alpha_{\mathrm{CID}} + \beta_{\mathrm{CID}} \log P
    \alpha ~ \mathcal{N}(3, 0.5)
    \beta ~ \mathcal{N}(0, 0.2)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/Kline.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10 entries, 0 to 9
# Data columns (total 5 columns):
#    Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   culture      10 non-null     object
#  1   population   10 non-null     int64
#  2   contact      10 non-null     object
#  3   total_tools  10 non-null     int64
#  4   mean_TU      10 non-null     float64
# dtypes: float64(1), int64(2), object(2)
# memory usage: 532.0 bytes

# (R code 11.40)
# allow model evaluation at arbitrary values
df['population'] = df['population'].astype(float)
df['P'] = sts.standardize(np.log(df['population']))
df['contact_id'] = (df['contact'] == 'high').astype(int)

# -----------------------------------------------------------------------------
#         Test Priors
# -----------------------------------------------------------------------------
# Test some priors to get a sense of the log scale
xs = np.linspace(0, 100, 200)

# (R code 11.41-43)
# NOTE r::dlnorm(x, meanlog, sdlog) -> s = sdlog, scale = np.exp(meanlog)
α_weak = stats.lognorm.pdf(xs, scale=np.exp(0), s=10)
α_strong = stats.lognorm.pdf(xs, scale=np.exp(3), s=0.5)

fig, ax = plt.subplots(num=1, clear=True)
ax.plot(xs, α_weak, 'k-', label=r'$\alpha \sim \mathcal{N}(0, 10)$')
ax.plot(xs, α_strong, 'C0-', label=r'$\alpha \sim \mathcal{N}(3, 0.5)$')

ax.legend()
ax.set(
    xlabel='Mean number of tools',
    ylabel=r'Density $e^\alpha$',
    xlim=(0, 100),
    ylim=(0, 0.08),
)
ax.spines[['top', 'right']].set_visible(False)

# -----------------------------------------------------------------------------
#         Plot prior predictive simulations
# -----------------------------------------------------------------------------
# (R code 11.44)
N_lines = 100

# Use standardized x like we would for modeling
xs = np.linspace(-2, 2, 200)[:, np.newaxis]
α = stats.norm(3, 0.5).rvs(N_lines)
β_weak = stats.norm(0, 10).rvs(N_lines)
prior_weak = np.exp(α + β_weak * xs)

β_strong = stats.norm(0, 0.2).rvs(N_lines)
prior_strong = np.exp(α + β_strong * xs)

# Creat unstandardized x data for log(population) (R code 11.45)
# This is the same as creating linear data and plotting on log scale
x = np.linspace(np.log(100), np.log(200_000), 200)[:, np.newaxis]
λ = np.exp(α + β_strong * x)

fig, axs = plt.subplots(num=2, nrows=2, ncols=2, sharey='row', clear=True)

axs[0, 0].plot(xs, prior_weak, 'k-', lw=1, alpha=0.4)
axs[0, 0].set(
    title=r'$\beta \sim \mathcal{N}(0, 10)$',
    xlabel='log population [std]',
    ylabel='total tools',
    xlim=(-2, 2),
    ylim=(0, 100),
)
axs[0, 0].set_xticks(np.arange(-2, 3))

axs[0, 1].plot(xs, prior_strong, 'k-', lw=1, alpha=0.4)
axs[0, 1].set(
    title=r'$\beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='log population [std]',
    xlim=(-2, 2),
)
axs[0, 1].set_xticks(np.arange(-2, 3))

# Plot un-standardized x-axis (R code 11.46)
axs[1, 0].plot(x, λ, 'k-', lw=1, alpha=0.4)
axs[1, 0].set(
    title=r'$\alpha \sim \mathcal{N}(3, 0.5), \beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='log population',
    ylabel='total tools',
    xlim=(x.min(), x.max()),
    ylim=(0, 500),
)

# Plot on standard population scale (R code 11.47)
axs[1, 1].plot(np.exp(x), λ, 'k-', lw=1, alpha=0.4)
axs[1, 1].set(
    title=r'$\alpha \sim \mathcal{N}(3, 0.5), \beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='population',
    ylabel='total tools',
    xlim=(0, 200_000),
    ylim=(0, 500),
)

# -----------------------------------------------------------------------------
#         Build the Models (R code 11.48)
# -----------------------------------------------------------------------------
# Intercept only
with pm.Model() as model:
    α = pm.Normal('α', 3, 0.5)
    λ = pm.Deterministic('λ', pm.math.exp(α))
    T = pm.Poisson('T', λ, observed=df['total_tools'])
    m11_9 = sts.ulam(data=df)

# Interaction model
with pm.Model() as model:
    P = pm.MutableData('P', df['P'])
    cid = pm.MutableData('cid', df['contact_id'])
    α = pm.Normal('α', 3, 0.5, shape=(2,))
    β = pm.Normal('β', 0, 0.2, shape=(2,))
    λ = pm.Deterministic('λ', pm.math.exp(α[cid] + β[cid]*P))
    T = pm.Poisson('T', λ, observed=df['total_tools'])
    m11_10 = sts.ulam(data=df)

# (R code 11.49)
cmp = sts.compare(
    [m11_9, m11_10],
    mnames=['m11.9', 'm11.10'],
    ic='LOOIC',
    args=dict(warn=True)
)
with pd.option_context('display.precision', 2):
    print(cmp['ct'])


# -----------------------------------------------------------------------------
#         Plot model results with Pareto k values (R code 11.50)
# -----------------------------------------------------------------------------
# Un-standardized values on real scale
Ns = 100
Pseq = np.linspace(-5, 3, Ns)  # standardized scale
pop_seq = np.exp(
    sts.unstandardize(
        Pseq,
        np.log(df['population'])
    )
)  # natural scale


def plot_data(ax, model, x, xv, log_scale=True, topK=0):
    """Plot the data and mean fit, sized by the Pareto k value.

    Parameters
    ----------
    ax : Axes
        The axes in which to make the plot.
    model : PostModel
        The model to evaluate.
    x : str
        The name of the column of x-data.
    xv : array_like
        The x-values over which the mean is evaluated.
    topK : int >= 0, optional
        The number of Pareto k values to label.

    Returns
    -------
    None
    """
    # Get the scaling with the original data
    def reset_data():
        pm.set_data(
            dict(P=df['P' if log_scale else 'population'],
                 cid=df['contact_id']),
            model=model.model
        )

    reset_data()
    loo = sts.LOOIS(model, pointwise=True)['T']
    pk = loo['pareto_k'] / loo['pareto_k'].max()

    pop = Pseq if log_scale else pop_seq

    # Closed circles for high contact, open circles for low contact
    ax.scatter(x, 'total_tools', data=df,
               ec='C0', fc=np.where(df['contact'] == 'high', 'C0', 'none'),
               s=1 + 10*np.exp(3*pk))

    for c, ls in zip([0, 1], ['--', '-']):
        # Sample the mean
        λ_samp = sts.lmeval(
            model,
            out=model.model.λ,
            eval_at=dict(P=pop, cid=np.full_like(pop, c, dtype=int)),
        )
        sts.lmplot(
            fit_x=xv, fit_y=λ_samp,
            ax=ax,
            label=('high' if c else 'low') + ' contact',
            line_kws=dict(c='k', ls=ls),
            fill_kws=dict(fc='k', alpha=0.1),
        )

    # Label top K
    if topK:
        idx = loo.sort_values('pareto_k', ascending=False).index[:topK]
        for i in idx:
            name, tx, ty = df.loc[i, ['culture', x, 'total_tools']]
            k = loo.loc[i, 'pareto_k']
            ax.text(s=f"{name} ({k:.2f})", x=tx, y=ty+2,
                    ha='center', va='bottom')

    ax.spines[['top', 'right']].set_visible(False)
    reset_data()
    return


# Make the plot
fig, axs = plt.subplots(num=3, ncols=2, sharey=True, clear=True)
fig.set_size_inches((10, 5), forward=True)

plot_data(axs[0], model=m11_10, x='P', xv=Pseq, topK=4)
plot_data(axs[1], model=m11_10, x='population', xv=pop_seq)

# TODO dummy legend entries for the open/closed circles. Move inside plot_data.
axs[0].legend(loc='upper left')
axs[0].set(
    xlabel='log population [std]',
    xlim=(-1.5, 3),
    ylabel='total tools',
    ylim=(None, 90),
)

axs[1].set(
    xlabel='population',
    ylabel='',
    xticks=[0, 50_000, 150_000, 250_000],
    xlim=(-10_000, 300_000),
)
axs[1].xaxis.set_major_formatter('{x:,d}')


# -----------------------------------------------------------------------------
#         Create "scientific" model
# -----------------------------------------------------------------------------
# (R code 11.52)
with pm.Model() as model:
    P = pm.MutableData('P', df['population'])
    cid = pm.MutableData('cid', df['contact_id'])
    α = pm.Normal('α', 1, 1, shape=(2,))
    β = pm.Exponential('β', 1, shape=(2,))
    γ = pm.Exponential('γ', 1)
    λ = pm.Deterministic('λ', pm.math.exp(α[cid]) * P**β[cid] / γ)
    T = pm.Poisson('T', λ, observed=df['total_tools'])
    m11_11 = sts.ulam(data=df)

# Subsume γ into α, since it is just a constant
with pm.Model() as model:
    P = pm.MutableData('P', df['population'])
    cid = pm.MutableData('cid', df['contact_id'])
    α = pm.Normal('α', 1, 1, shape=(2,))
    β = pm.Exponential('β', 1, shape=(2,))
    λ = pm.Deterministic('λ', pm.math.exp(α[cid]) * P**β[cid])
    T = pm.Poisson('T', λ, observed=df['total_tools'])
    m11_11x = sts.ulam(data=df)


# (R code 11.51)
fig, ax = plt.subplots(num=4, clear=True)
plot_data(ax, model=m11_11, log_scale=False, x='population', xv=pop_seq)

ax.legend(loc='lower right')
ax.set(
    xlabel='population',
    ylabel='total tools',
    xticks=[0, 50_000, 150_000, 250_000],
    xlim=(-10_000, 300_000),
    ylim=(0, 90),
)
ax.xaxis.set_major_formatter('{x:,d}')

print(sts.coef_table([m11_11, m11_11x], ['11', '11x'], hist=True))
print(sts.compare([m11_11, m11_11x], ['11', '11x'])['ct'])


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
