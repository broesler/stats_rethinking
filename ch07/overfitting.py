#!/usr/bin/env python3
# =============================================================================
#     File: overfitting.py
#  Created: 2023-05-17 22:42
#   Author: Bernie Roesler
#
"""
§7.1 The Problem with Parameters.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats
from tqdm import tqdm

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

SIGMA_CONST = 0.001

# Manually define data (R code 7.1)
sppnames = ['afarensis', 'africanus', 'habilis', 'boisei', 'rudolfensis',
            'ergaster', 'sapiens']
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

df = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))

# Standardize variables (R code 7.2)
df['mass_std'] = sts.standardize(df['mass'])
df['brain_std'] = df['brain'] / df['brain'].max()  # [0, 1]

# Use vague priors (R code 7.3)
with pm.Model():
    α = pm.Normal('α', 0.5, 1)
    β = pm.Normal('β', 0, 10)
    μ = pm.Deterministic('μ', α + β * df['mass_std'])
    log_σ = pm.Normal('log_σ', 0, 1)
    brain_std = pm.Normal('brain_std', μ, pm.math.exp(log_σ),
                          observed=df['brain_std'])
    m7_1 = sts.quap(data=df)

print('m7.1:')
sts.precis(m7_1)

# Compute R² value (R code 7.4)
h_samp = sts.lmeval(
    m7_1,
    out=m7_1.model.brain_std,
    params=[m7_1.model.α,
            m7_1.model.β,
            m7_1.model.log_σ],
)
h_mean = h_samp.mean(axis=1)
r = h_samp.mean(axis=1) - df['brain_std']  # residuals
Rsq = 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)
print(f"{Rsq = :.4f}")

# Plot the data and linear predictions
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter('mass', 'brain', data=df)
for label, x, y in zip(df['species'], df['mass'], df['brain']):
    ax.text(x+0.5, y+2, label)
idx = np.argsort(df['mass']).values
ax.plot(df.loc[idx, 'mass'], h_mean[idx] * df['brain'].max(), c='k', marker='x')
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')


# -----------------------------------------------------------------------------
#         Create polynomial models (R code 7.6 - 7.8)
# -----------------------------------------------------------------------------
models = dict()
Rsqs = dict()
Np = 6  # max polynomial terms


# (R code 7.5)
def brain_Rsq(quap):
    """Compute the :math:`R^2` value of the model."""
    params = [quap.model.α]
    var_names = quap.model.named_vars.keys()
    if 'βn' in var_names:
        params.append(quap.model.βn)
    if 'log_σ' in var_names:
        params.append(quap.model.log_σ)
    h_samp = sts.lmeval(quap, out=quap.model.brain_std, params=params)
    r = h_samp.mean(axis=1) - df['brain_std']  # residuals
    # pandas default is ddof=1 => N-1, so explicitly use ddof=0 => N.
    return 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)


def poly_model(poly_order, x='mass_std', y='brain_std', data=df):
    """Build a polynomial model of the brain ~ mass relationship."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        X = sts.design_matrix(ind, poly_order)  # [1 x x² x³ ...]
        α = pm.Normal('α', 0.5, 1, shape=(1,))
        βn = pm.Normal('βn', 0, 10, shape=(poly_order,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        if poly_order < 6:
            log_σ = pm.Normal('log_σ', 0, 1)
            sigma = pm.math.exp(log_σ)
        else:
            sigma = SIGMA_CONST
        brain_std = pm.Normal('brain_std', μ, sigma,
                              observed=data[y], shape=ind.shape)
        # Compute the posterior
        quap = sts.quap(data=data)
    return quap


# Plot each polynomial fit (Figure 7.3, see plotting_support.R -> brain_plot)
fig = plt.figure(2, clear=True, constrained_layout=True)
fig.set_size_inches((8, 10), forward=True)
gs = fig.add_gridspec(nrows=3, ncols=2)
xe_s = np.linspace(df['mass_std'].min() - 0.2, df['mass_std'].max() + 0.2, 200)

for poly_order in range(1, Np+1):
    # Build the model and fit the posterior
    quap = poly_model(poly_order)

    # Store and print the models and R² values
    k = f"m7.{poly_order}"
    models[k] = quap
    Rsqs[k] = brain_Rsq(quap)
    print(k)
    sts.precis(quap)
    print(f"R² = {Rsqs[k]:.4f}")

    # Plot the fit
    i = poly_order - 1
    sharex = ax if i > 0 else None
    ax = fig.add_subplot(gs[i], sharex=sharex)

    # Sample the posterior manually and explicitly
    mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                         params=[quap.model.α, quap.model.βn])
    # Re-scale the variables
    mu_samp *= df['brain'].max()
    xe = sts.unstandardize(xe_s, df['mass'])

    # PLot results
    sts.lmplot(fit_x=xe, fit_y=mu_samp,
               x='mass', y='brain', data=df,
               ax=ax,
               line_kws=dict(c='k', lw=1),
               fill_kws=dict(facecolor='k', alpha=0.2))

    ax.set_title(rf"{k}: $R^2 = {Rsqs[k]:.2f}$",
                 x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='body mass [kg]',
           ylabel='brain volume [cc]')

    if i < 4:  # all except last row
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=None)
        ax.set_ylim((300, 1500))
    elif i == 4:
        ax.set_ylim((0, 2100))
    elif i == 5:
        ax.set_ylim((-500, 2100))
        ax.axhline(0, c='k', lw=1, ls='--')


# -----------------------------------------------------------------------------
#         Underfitting (R code 7.11)
# -----------------------------------------------------------------------------
with pm.Model():
    ind = pm.MutableData('ind', df['mass_std'])
    α = pm.Normal('α', 0.5, 1)
    μ = pm.Deterministic('μ', α * pm.math.ones_like(ind))  # NOT function of x!
    log_σ = pm.Normal('log_σ', 0, 1)
    brain_std = pm.Normal('brain_std', μ, pm.math.exp(log_σ),
                          observed=df['brain_std'], shape=ind.shape)
    # Compute the posterior
    m7_7 = sts.quap(data=df)

# Store and print the models and R² values
k = "m7.7"
models[k] = m7_7
Rsqs[k] = brain_Rsq(m7_7)

print(k)
sts.precis(m7_7)
print(f"R² = {Rsqs[k]:.4f}")

# Figure 7.4
fig = plt.figure(3, clear=True, constrained_layout=True)
ax = fig.add_subplot()

# Re-scale the variables
mu_samp = sts.lmeval(m7_7, out=m7_7.model.μ, eval_at={'ind': xe_s},
                     params=[m7_7.model.α])
mu_samp *= df['brain'].max()
xe = sts.unstandardize(xe_s, df['mass'])

# PLot results
sts.lmplot(fit_x=xe, fit_y=mu_samp,
           x='mass', y='brain', data=df,
           ax=ax,
           line_kws=dict(c='k', lw=1),
           fill_kws=dict(facecolor='k', alpha=0.2))

ax.set_title(rf"{k}: $R^2 = {Rsqs[k]:.2f}$", x=0.02, y=1, loc='left', pad=-14)
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')


# -----------------------------------------------------------------------------
#         Variance -- Leave-one-out validation
# -----------------------------------------------------------------------------
# Figure 7.5
# Re-fit N=1 and N=4 polynomials to the data, leaving one data point out.

fig = plt.figure(4, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)

poly_orders = [1, 4]
for i, poly_order in tqdm(enumerate(poly_orders),
                          total=len(poly_orders),
                          position=0,
                          desc='poly_order'):
    ax = fig.add_subplot(gs[i])
    # Fit to the data - 1 row at a time
    for j in tqdm(range(len(df)),
                  position=1,
                  desc='data pts',
                  leave=False):
        # Create the model
        quap = poly_model(poly_order, x='mass_std', y='brain_std',
                          data=df.drop(j))
        mu_samp = sts.lmeval(quap, out=quap.model.μ, eval_at={'ind': xe_s},
                             params=[quap.model.α, quap.model.βn])
        mu_mean = mu_samp.mean(axis=1) * df['brain'].max()
        ax.plot(xe, mu_mean, 'k', lw=1.5, alpha=0.4)

    # Plot the results together
    ax.scatter('mass', 'brain', data=df)
    if poly_order == 4:
        ax.set_ylim((-200, 2200))
    ax.set_title(f"m7.{poly_order}", x=0.02, y=1, loc='left', pad=-14)
    ax.set(xlabel='body mass [kg]',
           ylabel='brain volume [cc]')

# -----------------------------------------------------------------------------
#         §7.2 Log Pointwise Predictive Density (lppd)
# -----------------------------------------------------------------------------
# Compute the log probabilities of the model (R code 7.14, 7.15)


def lppd(quap, Ns=1000):
    """Compute the log pointwise predictive density for a model."""
    Ns = int(Ns)
    post = quap.sample(Ns)
    mu_samp = sts.lmeval(
            quap,
            out=quap.model.μ,
            dist=post,
            eval_at={'ind': df['mass_std']}
        )
    sigma = np.exp(post['log_σ']) if 'log_σ' in post else SIGMA_CONST
    h_logp = stats.norm(mu_samp, sigma).logpdf(df[['brain_std']])
    return sts.logsumexp(h_logp, axis=1) - np.log(Ns)


# R code 7.14
# R> lppd( m7.1 , n=1e4 )
# [1]  0.6099  0.6483  0.5496  0.6235  0.4648  0.4348 -0.8445
#
# >>> lppd(models['m7.1'])
# === array([ 0.6156,  0.6508,  0.5414,  0.6316,  0.4698,  0.4349, -0.8536])
# OR:
# >>> sts.lppd(m7_1)['brain_std']
# === array([ 0.6286,  0.6678,  0.5485,  0.6373,  0.4579,  0.4187, -0.8548])


# R code 7.16
print('lppd:')
SEED = 1
np.random.seed(SEED)
the_lppds = pd.DataFrame({k: lppd(v) for k, v in models.items()})
the_lppds.loc['Sum', :] = the_lppds.sum(axis='rows')
print(the_lppds.loc['Sum'])

# Compute WAIC and LOOIS to check variation with number of parameters
np.random.seed(SEED)
the_waics = pd.DataFrame({k: sts.WAIC(v)['brain_std'] for k, v in models.items()})

R_waics = pd.DataFrame(
    data={
        'WAIC':    [6.821, 10   , 11.25, 15.56, -0.04896, -71.38, 5.376],
        'lppd':    [2.49 , 2.566, 3.707, 5.334, 14.11   , 39.45 , 0.3619],
        'penalty': [5.901, 7.568, 9.331, 13.11, 14.08   , 3.756 , 3.05],
        'std':     [9.67 , 8.092, 9.046, 6.493, 3.616   , 0.1714, 6.685],
    },
    index=the_waics.columns,
).T

np.random.seed(SEED)
the_loos = pd.DataFrame(
    {k: sts.LOOIS(v)['brain_std'] for k, v in models.items()}
)

R_loos = pd.DataFrame(
    data={
        'PSIS':    [17.6  , 29.36 , 34.71 , 55.3  , 55.7  , -68.68, 9.802],
        'lppd':    [-8.802, -14.68, -17.36, -27.65, -27.85, 34.34 , -4.901],
        'penalty': [11.29 , 17.24 , 21.06 , 32.98 , 41.96 , 5.104 , 5.263],
        'std':     [19.65 , 17.78 , 18.58 , 16.76 , 7.355 , 0.4465, 11.56],
    },
    index=the_waics.columns
).T

# ----------------------------------------------------------------------------- 
#         Plot the criteria vs deviance
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(ncols=2, num=5, clear=True, constrained_layout=True)

# Sort data by the number of parameters
params = np.r_[range(1, 7), 0]
pdx = np.argsort(params)
params = params[pdx]

the_lppds = the_lppds.iloc(axis=1)[pdx]
the_waics = the_waics.iloc(axis=1)[pdx]
the_loos = the_loos.iloc(axis=1)[pdx]
R_waics = R_waics.iloc(axis=1)[pdx]
R_loos = R_loos.iloc(axis=1)[pdx]

deviance = -2 * the_lppds.loc['Sum']

# R> the_devs <- sapply(
#   list(m7.7, m7.1, m7.2, m7.3, m7.4, m7.5, m7.6) ,
#   function(m) -2 * sum(lppd(m)) 
# )
R_dev = np.r_[-0.7792, -4.9616, -5.2139, -7.4706, -10.6454, -27.9447, -78.9699]

ax = axes[0]
ax.plot(params, deviance, c='C0', label='py deviance')
ax.plot(params, R_dev, 'k--', label='R deviance')

ax.scatter(params, the_waics.loc['WAIC'], c='C0', label='py WAIC')
ax.scatter(params, R_waics.loc['WAIC'], edgecolor='C0', facecolor='none', label='R WAIC')

ax.scatter(params, the_loos.loc['PSIS'], c='k', label='py LOOIS')
ax.scatter(params, R_loos.loc['PSIS'], edgecolor='k', facecolor='none', label='R LOOIS')

ax.set_xlabel('# of parameters')
ax.legend()

ax = axes[1]
ax.scatter(params, (deviance - R_dev) / deviance, edgecolor='k', facecolor='none', label='deviance')
ax.scatter(params, (the_waics.loc['WAIC'] - R_waics.loc['WAIC']) / the_waics.loc['WAIC'], c='C0', label='WAIC')
ax.scatter(params, (the_loos.loc['PSIS'] - R_loos.loc['PSIS']) / the_loos.loc['PSIS'], c='k', label='LOOIS')

ax.set(xlabel='# of parameters',
       ylabel='py - R [% err]')
ax.legend()

# ----------------------------------------------------------------------------- 
#         Compare to R versions
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
#         LPPD
# -----------------------------------------------------------------------------
# >>> print(the_lppds.loc['Sum'])
# m7.1     2.482847
# m7.2     2.583188
# m7.3     3.628317
# m7.4     5.337486
# m7.5    14.124919
# m7.6    39.540737
# m7.7     0.355546
# Name: Sum, dtype: float64

# R> sapply( list(m7.1, m7.2, m7.3, m7.4, m7.5, m7.6, m7.7) , function(m) sum(lppd(m)) )
# [1]  2.453  2.620  3.719  5.338 14.053 39.554  0.385

# ----------------------------------------------------------------------------- 
#         WAIC
# -----------------------------------------------------------------------------
# >>> the_waics
# ===
#              m7.1      m7.2       m7.3       m7.4       m7.5       m7.6      m7.7
# waic     3.964978  8.436108  11.182932  13.854514  -1.292793 -72.061388  6.139253
# lppd     2.495198  2.628192   3.651979   5.328711  14.001523  39.541272  0.361894
# penalty  4.477687  6.846246   9.243445  12.255969  13.355126   3.510578  3.431521
# std      7.402316  7.021514   9.390872   7.025531   6.234243   0.271815  7.321834

# R> set.seed(1)
# R> sapply( list(m7.1, m7.2, m7.3, m7.4, m7.5, m7.6, m7.7) , function(m) WAIC(m) )
#         [,1]  [,2]  [,3]  [,4]  [,5]     [,6]   [,7]
# WAIC    6.821 10    11.25 15.56 -0.04896 -71.38 5.376
# lppd    2.49  2.566 3.707 5.334 14.11    39.45  0.3619
# penalty 5.901 7.568 9.331 13.11 14.08    3.756  3.05
# std_err 9.67  8.092 9.046 6.493 3.616    0.1714 6.685

# ----------------------------------------------------------------------------- 
#         LOOIS
# -----------------------------------------------------------------------------
# >>> the_loos
# ===
#               m7.1       m7.2       m7.3       m7.4       m7.5       m7.6       m7.7
# PSIS     10.475741  19.331760  33.701941  47.261409  44.966345 -68.971715  13.449618
# lppd     -5.237870  -9.665880 -16.850971 -23.630705 -22.483173  34.485857  -6.724809
# penalty   7.733069  12.294072  20.502950  28.959416  36.484696   5.055414   7.086703
# std      11.512229  10.328729  20.815729  13.641037  14.543865   0.696069  13.987098

# R> sapply( list(m7.1, m7.2, m7.3, m7.4, m7.5, m7.6, m7.7) , function(m) LOO(m, warn=FALSE) )
#         [,1]   [,2]   [,3]   [,4]   [,5]   [,6]   [,7]
# PSIS    17.6   29.36  34.71  55.3   55.7   -68.68 9.802
# lppd    -8.802 -14.68 -17.36 -27.65 -27.85 34.34  -4.901
# penalty 11.29  17.24  21.06  32.98  41.96  5.104  5.263
# std_err 19.65  17.78  18.58  16.76  7.355  0.4465 11.56

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
