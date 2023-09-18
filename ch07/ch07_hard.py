#!/usr/bin/env python3
# =============================================================================
#     File: ch07_hard.py
#  Created: 2023-08-30 15:39
#   Author: Bernie Roesler
#
"""
Hard exercises from Ch. 7.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

df = pd.read_csv('../data/Howell1.csv')

# >>> df.info
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 544 entries, 0 to 543
# Data columns (total 4 columns):
#    Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   height  544 non-null    float64
#  1   weight  544 non-null    float64
#  2   age     544 non-null    float64
#  3   male    544 non-null    int64
# dtypes: float64(3), int64(1)
# memory usage: 17.1 KB

df['A'] = sts.standardize(df['age'])

train = df.sample(frac=0.5, random_state=1000)  # "d1" in the book
test = df.drop(train.index)                     # "d2"

assert len(train) == 272

# Plot the data
fig, ax = plt.subplots(num=1, clear=True, constrained_layout=True)
ax.scatter('A', 'height', data=df, alpha=0.4)
ax.set(xlabel='age [std]',
       ylabel='height [cm]')


def poly_model(poly_order, x='A', y='height', data=train, priors='weak'):
    r"""Build a polynomial model of the height ~ age relationship.

    The model takes the form:

    .. math::
        h_i \sim \mathcal{N}(\mu_i, \sigma)
        \mu_i = \alpha + \beta_1 x_i + \beta_2 x_i^2 + \dots + \beta_N x_i^N.


    Parameters
    ----------
    poly_order : int
        Order of the polynomial.
    x, y : str
        Names of the data columns to use for the model input/output.
    data : pd.DataFrame
        The input/output data.
    priors : str in {'flat', 'weak'}
        Choice of priors.

    Returns
    -------
    quap : :obj:`sts.Quap`
        The quadratic approximation model fit.
    """
    with pm.Model():
        # Set the data input to the model
        ind = pm.MutableData('ind', data[x])
        X = sts.design_matrix(ind, poly_order)  # [1 x x² x³ ...]
        # Define the priors
        match priors:
            case 'flat':
                # Select flat priors
                α_mean = data[y].mean()
                α_std = 10*data[y].std()
                βn_std = 1000
            case 'weak':
                # Select "weakly" informative priors
                α_mean = data[y].mean()
                α_std = data[y].std()
                βn_std = 100
            case _:
                raise ValueError(f"priors={priors} is unsupported.")
        # Define the model parameters
        α = pm.Normal('α', α_mean, α_std, shape=(1,))
        βn = pm.Normal('βn', 0, βn_std, shape=(poly_order,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        σ = pm.LogNormal('σ', 0, 1)
        h = pm.Normal('h', μ, σ, observed=data[y], shape=ind.shape)
        # Compute the quadratic posterior approximation
        quap = sts.quap(data=data)
    return quap


# Create polynomial models of height ~ age.
Np = 6  # max polynomial terms
models = {i: poly_model(i, data=train, priors='weak') for i in range(1, Np+1)}

# Prior predictive checks with the linear model
N = 20
with models[1].model:
    idata = pm.sample_prior_predictive(N)

fig, ax = plt.subplots(num=2, clear=True, constrained_layout=True)
ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
ax.plot(models[1].data['A'], idata.prior['μ'].mean('chain').T, 'k', alpha=0.4)
ax.set(xlabel='age [std]',
       ylabel='height [cm]')


# -----------------------------------------------------------------------------
#         6H1: Compare the models using WAIC
# -----------------------------------------------------------------------------
cmp = sts.compare(models.values(), mnames=models.keys())
ct = cmp['ct']
print(ct)
fig, ax = sts.plot_compare(ct, fignum=3)
ax.set_xlabel('Polynomial order')

# -----------------------------------------------------------------------------
#         6H2: Plot each model mean and CI
# -----------------------------------------------------------------------------
fig = plt.figure(4, clear=True, constrained_layout=True)
fig.set_size_inches((8, 10), forward=True)
gs = fig.add_gridspec(nrows=3, ncols=2)
xe_s = np.linspace(df['A'].min() - 0.2, df['A'].max() + 0.2, 200)

for poly_order in range(1, Np+1):
    # Get the model
    quap = models[poly_order]

    # Store and print the models and R² values
    print(f"Model {poly_order}:")
    sts.precis(quap)

    # Plot the fit
    i = poly_order - 1
    sharex = ax if i > 0 else None
    ax = fig.add_subplot(gs[i], sharex=sharex)

    # Sample the posterior manually and explicitly
    mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                         params=[quap.model.α, quap.model.βn])
    # Re-scale the input
    xe = sts.unstandardize(xe_s, df['age'])

    # PLot results
    sts.lmplot(fit_x=xe, fit_y=mu_samp,
               x='age', y='height', data=df,
               ax=ax,
               q=0.97,  # 97% confidence intervals
               line_kws=dict(c='k', lw=1),
               fill_kws=dict(facecolor='k', alpha=0.2))

    ax.set_title(rf"$\mathcal{{M}}_{poly_order}$",
                 x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='age [yrs]',
           ylabel='height [cm]')


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
