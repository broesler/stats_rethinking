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
import xarray as xr

from tqdm import tqdm

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
        obs = pm.MutableData('obs', data[y])
        X = sts.design_matrix(ind, poly_order)  # [1 x x² x³ ...]
        # Define the priors
        match priors:
            case 'flat':
                # Select flat priors
                α_mean = data[y].mean()
                α_std = 10*data[y].std()
                β_std = 1000
            case 'weak':
                # Select "weakly" informative priors
                α_mean = data[y].mean()
                α_std = data[y].std()
                β_std = 100
            case 'strong':
                # Select "stronger" informative priors
                α_mean = data[y].mean()
                α_std = data[y].std()
                β_std = 5
            case _:
                raise ValueError(f"priors='{priors}' is unsupported.")
        # Define the model parameters
        α = pm.Normal('α', α_mean, α_std, shape=(1,))
        βn = pm.Normal('βn', 0, β_std, shape=(poly_order,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        σ = pm.LogNormal('σ', 0, 1)
        h = pm.Normal('h', μ, σ, observed=obs, shape=ind.shape)
        # Compute the quadratic posterior approximation
        quap = sts.quap(data=data)
    return quap


# Create polynomial models of height ~ age.
Np = 6  # max polynomial terms
print('Fitting models... ', end='')
models = {i: poly_model(i, data=train, priors='weak') 
          for i in tqdm(range(1, Np+1))}
print('done.')

# Prior predictive checks with the linear model
N = 20
prior = models[1].sample_prior(N)

fig, ax = plt.subplots(num=2, clear=True, constrained_layout=True)
ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
ax.plot(models[1].data['A'], prior['μ'].T, 'k', alpha=0.4)
ax.set(xlabel='age [std]',
       ylabel='height [cm]')


# -----------------------------------------------------------------------------
#         7H1: Compare the models using WAIC
# -----------------------------------------------------------------------------
cmp = sts.compare(models.values(), mnames=models.keys())
ct = cmp['ct']
print(ct)
#               WAIC         SE       dWAIC        dSE   penalty         weight
# model
# 1      2416.562108  21.805795  505.021401  28.052322  3.396792  1.713077e-110
# 2      2155.272143  23.761716  243.731437  26.032288  5.036507   9.379328e-54
# 3      1942.544720  21.267677   31.004013  14.024238  5.733186   1.463338e-07
# 4      1911.540706  22.381670    0.000000        NaN  6.438159   7.902795e-01
# 5      1914.731506  22.741761    3.190800   0.950519  7.991992   1.602903e-01
# 6      1917.084364  23.189859    5.543657   3.699959  9.916867   4.943003e-02

fig, ax = sts.plot_compare(ct, fignum=3)
ax.set_xlabel('Polynomial order')

ct = ct.xs('h')  # only one variable anyway

# -----------------------------------------------------------------------------
#         7H2: Plot each model mean and CI
# -----------------------------------------------------------------------------
fig = plt.figure(4, clear=True, constrained_layout=True)
fig.set_size_inches((8, 10), forward=True)
gs = fig.add_gridspec(nrows=3, ncols=2)

# Evaluate the oodel over the entire range of data, plot unstandardized
xe_s = np.linspace(df['A'].min() - 0.2, df['A'].max() + 0.2, 100)
xe = sts.unstandardize(xe_s, df['age'])

mean_samples = xr.Dataset()

for poly_order in range(1, Np+1):
    # Get the model
    quap = models[poly_order]

    # Store and print the models and R² values
    print(f"\nModel {poly_order}:")
    sts.precis(quap)

    # Plot the fit
    i = poly_order - 1
    sharex = ax if i > 0 else None
    ax = fig.add_subplot(gs[i], sharex=sharex)

    # Sample the posterior manually and explicitly
    # Need this step because lmplot cannot unstandardize just one of the
    # dimensions at a time.
    mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                         params=[quap.model.α, quap.model.βn])
    mean_samples[poly_order] = mu_samp

    # Plot results
    sts.lmplot(fit_x=xe, fit_y=mu_samp,
               x='age', y='height', data=df,
               q=0.97,  # 97% confidence intervals
               line_kws=dict(c='k', lw=1),
               fill_kws=dict(facecolor='k', alpha=0.2),
               ax=ax)

    ax.set_title(rf"$\mathcal{{M}}_{poly_order}$",
                 x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='age [yrs]',
           ylabel='height [cm]')


# -----------------------------------------------------------------------------
#         7H3: Plot the model averaged predictions
# -----------------------------------------------------------------------------
# Take mu_samp from above and weight it by ct['weight']
mean_samples = mean_samples.to_array(dim='poly_order')  # [Np, len(xe), Ns)
weights = np.reshape(ct['weight'], (-1, 1, 1))  # (Np, 1, 1)

# weights are normalized to 1
weighted_model = (mean_samples * weights).sum('poly_order')

fig, ax = plt.subplots(num=5, clear=True, constrained_layout=True)

# Plot the minimum WAIC model (model 4)
p_best = ct['WAIC'].idxmin()
quap = models[int(p_best)]
mu_samp_best = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                          params=[quap.model.α, quap.model.βn])

sts.lmplot(fit_x=xe, fit_y=mu_samp_best,
           q=0.97,
           line_kws=dict(c='k'),
           fill_kws=dict(facecolor='k'),
           label=rf"$\mathcal{{M}}_{p_best}$",
           ax=ax)

# Plot the weighted means vs the data
sts.lmplot(fit_x=xe, fit_y=weighted_model,
           x='age', y='height', data=df,
           q=0.97,  # 97% confidence intervals
           line_kws=dict(c='C1'),
           fill_kws=dict(facecolor='C1'),
           label='Model-Averaged',
           ax=ax)

ax.set(title='Model-averaged Predictions',
       xlabel='age [yrs]',
       ylabel='height [cm]')
ax.legend()

# NOTE that model-averaged predictions are not terribly different from M_4 in
# the mean, but have a tighter confidence bound than the single model because
# M_1, M_2, and M_3 have weights ≈ 0.


# -----------------------------------------------------------------------------
#         7H4: Compute the test-sample deviance
# -----------------------------------------------------------------------------
dev_test = pd.Series({
    str(p): sts.deviance(
        models[p],
        eval_at={
            'ind': test['A'],
            'obs': test['height']
            }
    )['h']

    for p in range(1, Np+1)
})

dev_test.name = 'deviance'
dev_test.index.name = 'model'

# -----------------------------------------------------------------------------
#         7H5: Compare the deviances to the WAIC values
# -----------------------------------------------------------------------------
cmp_test = pd.concat([dev_test, ct['WAIC']], axis='columns')
cmp_test['dev - WAIC'] = cmp_test['deviance'] - cmp_test['WAIC']
cmp_test['dWAIC'] = cmp_test['WAIC'] - cmp_test['WAIC'].min()
cmp_test['ddev'] = cmp_test['deviance'] - cmp_test['deviance'].min()

print('\nCompare deviance to WAIC:')
print(cmp_test)
#           deviance         WAIC  dev - WAIC       dWAIC        ddev
# model
# 1      2409.879794  2416.562108   -6.682314  505.021401  512.312787
# 2      2144.954603  2155.272143  -10.317540  243.731437  247.387596
# 3      1931.396127  1942.544720  -11.148593   31.004013   33.829120
# 4      1898.642917  1911.540706  -12.897790    0.000000    1.075910
# 5      1898.929806  1914.731506  -15.801700    3.190800    1.362799
# 6      1897.567007  1917.084364  -19.517357    5.543657    0.000000

# NOTE WAIC does a good job of approximating the actual out-of-sample deviance.
# It is increasingly different with higher numbers of parameters, however. The
# model that actually makes the best predictions is that with the highest
# number of parameters, as expected, although the models with {4, 5, 6}
# parameters are quite similar.

# -----------------------------------------------------------------------------
#         7H6: Stronger regularizing priors
# -----------------------------------------------------------------------------
model_strong = poly_model(6, data=train, priors='strong')
print('\nStrongly regularizing priors:')
sts.precis(model_strong)

fig, ax = plt.subplots(num=6, clear=True, constrained_layout=True)

# Plot the minimum WAIC model (model 4)
sts.lmplot(fit_x=xe, fit_y=mu_samp_best,
           x='age', y='height', data=df,
           q=0.97,
           line_kws=dict(c='k'),
           fill_kws=dict(facecolor='k'),
           label=rf"$\mathcal{{M}}_{p_best}$",
           ax=ax)

# Plot the strongly regularized model
quap = model_strong
mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                     params=[quap.model.α, quap.model.βn])
sts.lmplot(fit_x=xe, fit_y=mu_samp,
           q=0.97,
           line_kws=dict(c='C2'),
           fill_kws=dict(facecolor='C2'),
           label='strongly regularized',
           ax=ax)

ax.set(title='Strongly Regularizing Priors',
       xlabel='age [yrs]',
       ylabel='height [cm]')
ax.legend()

# Compute the actual test deviance of the strongly regularized model
dev_strong = sts.deviance(
    model_strong,
    eval_at={
        'ind': test['A'],
        'obs': test['height']
    }
)['h']

print(f"\n{dev_strong - dev_test[p_best] = :.2f}")  # ≈ 7.00

# NOTE The regularized deviance is only a bit worse than the best WAIC model
# from before. It is enough to differentiate the model from the


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
