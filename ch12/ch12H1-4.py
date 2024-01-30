#!/usr/bin/env python3
# =============================================================================
#     File: 12H1-4.py
#  Created: 2024-01-24 18:50
#   Author: Bernie Roesler
#
"""
Exercises 12H1 through 12H4 (mislabeled in 2ed as 11H1 etc.)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(Path('../data/hurricanes.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 92 entries, 0 to 91
# Data columns (total 8 columns):
#    Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   name          92 non-null     object
#  1   year          92 non-null     int64
#  2   deaths        92 non-null     int64
#  3   category      92 non-null     int64
#  4   min_pressure  92 non-null     int64    # [millibar]
#  5   damage_norm   92 non-null     int64    # [$]
#  6   female        92 non-null     int64    # [bool]
#  7   femininity    92 non-null     float64  # [1..11] 1 == M, 11 == F
# dtypes: float64(1), int64(6), object(1)
# memory usage: 5.9 KB

# sns.pairplot(df)  # set deaths, damage_norm to log scale -> linear fit

fem_levels = np.arange(1, 12)

# -----------------------------------------------------------------------------
#         12H1. Simple Poisson Model
# -----------------------------------------------------------------------------
# Build a simple Poisson model of deaths
with pm.Model():
    α = pm.Normal('α', 3, 0.5)
    λ = pm.Deterministic('λ', pm.math.exp(α))
    D = pm.Poisson('D', λ, observed=df['deaths'])
    mD = sts.ulam(data=df)

print('Simple model:')
sts.precis(mD)

# λ = np.exp(mD.coef['α']) = 20.658 ~ df['deaths'].mean() = 20.652

# Build a model with femininity as a predictor
with pm.Model():
    F = pm.MutableData('F', df['femininity'])
    α = pm.Normal('α', 3, 0.5)
    β_F = pm.Normal('β_F', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β_F*F))
    D = pm.Poisson('D', λ, shape=λ.shape, observed=df['deaths'])
    mF = sts.ulam(data=df)

print('Femininity model:')
sts.precis(mF)

# Q: How strong is the association between femininity of name and deaths?
# A: Not strong.

# Q: Which storms does the model fit (retrodict) well?
# A: Storms with mean # of deaths

# Q: Which storms does the model fit (retrodict) poorly?
# A: Storms with large deviations from the mean # of deaths

fig, ax = plt.subplots(num=1, clear=True)
ax.scatter('femininity', 'deaths', data=df, alpha=0.4)
ax.axhline(df['deaths'].mean(), ls='--', lw=1, c='gray', label='mean deaths')

# Plot posterior mean
sts.lmplot(
    mF,
    mean_var=mF.model.λ,
    eval_at=dict(F=np.linspace(1, 11)),
    ax=ax
)

# Plot posterior predictive
sts.lmplot(
    mF,
    mean_var=mF.model.D,
    eval_at=dict(F=np.linspace(1, 11)),
    ax=ax
)

# -----------------------------------------------------------------------------
#         12H2. Gamma-Poisson Model
# -----------------------------------------------------------------------------
with pm.Model():
    F = pm.MutableData('F', df['femininity'])
    α = pm.Normal('α', 3, 0.5)
    β_F = pm.Normal('β_F', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β_F*F))
    φ = pm.Exponential('φ', 1)
    D = pm.NegativeBinomial('D', mu=λ, alpha=φ, shape=λ.shape,
                            observed=df['deaths'])
    mGF = sts.ulam(data=df)


# Plot posterior mean
sts.lmplot(
    mGF,
    mean_var=mGF.model.λ,
    eval_at=dict(F=np.linspace(1, 11)),
    line_kws=dict(c='C3'),
    fill_kws=dict(fc='C3'),
    ax=ax
)

# Plot posterior predictive
sts.lmplot(
    mGF,
    mean_var=mGF.model.D,
    eval_at=dict(F=np.linspace(1, 11)),
    line_kws=dict(c='none'),
    fill_kws=dict(fc='C3'),
    ax=ax
)

ax.set(xlabel='femininity',
       ylabel='deaths',
       xticks=fem_levels,
       yscale='log')

ax.legend(
    [plt.Line2D([0], [0], color='C0'),
     plt.Line2D([0], [0], color='C3')],
    ['Poisson Model', 'Gamma-Poisson Model']
)

# Q: Can you explain why the association diminished in strength?
# A: The Gamma-Poisson model has an 89% confidence interval that overlaps
# 0 deaths, as opposed to the Poisson model, which has an 89% confidence
# interval that barely drops below 9 deaths. Since nearly 2/3rds of the
# hurricanes have <= 10 deaths, the Poisson model

# -----------------------------------------------------------------------------
#         12H3. Include an interaction effect
# -----------------------------------------------------------------------------
# Normalize inputs
df['A'] = sts.standardize(df['damage_norm'])
df['P'] = sts.standardize(df['min_pressure'])

# Assumed DAG:
# F -> D <- A
#      ^  /
#      |/
#      P
#
# P -> A -> D is a pipe. Control for A to close the pipe.


def make_interaction_model(interaction, verbose=True):
    """Define a Ulam model with various interactions between F, A, and P.

    Parameters
    ----------
    interaction : str
        'A'         == D ~ F + A
        'P'         == D ~ F + P
        'A + P'     == D ~ F + A + P
        'F*A'       == D ~ F + A + F*A
        'F*P'       == D ~ F + P + F*P
        'F*A + F*P' == D ~ F + A + P + F*A + F*P
    verbose : bool, optional
        If True, print status.

    Returns
    -------
    result : Ulam
        The fitted model object.
    """
    if verbose:
        print(f"\n\n---------- Making model with {interaction = }...")

    with pm.Model():
        # Define the data
        F = pm.MutableData('F', df['femininity'])
        A = pm.MutableData('A', df['A'])
        P = pm.MutableData('P', df['P'])
        # Define the priors
        α = pm.Normal('α', 3, 0.5)
        β_F = pm.Normal('β_F', 0, 0.1)

        # Define the linear model
        linear_model = α + β_F*F

        if 'A' in interaction:
            β_A = pm.Normal('β_A', 0, 0.1)
            linear_model += β_A*A
        if 'P' in interaction:
            β_P = pm.Normal('β_P', 0, 0.1)
            linear_model += β_P*P

        # Add interaction terms
        if 'F*A' in interaction:
            β_FA = pm.Normal('β_FA', 0, 0.1)
            linear_model += β_FA*F*A
        if 'F*P' in interaction:
            β_FP = pm.Normal('β_FP', 0, 0.1)
            linear_model += β_FP*F*P

        # Build the outcome variable
        λ = pm.Deterministic('λ', pm.math.exp(linear_model))
        φ = pm.Exponential('φ', 1)
        D = pm.NegativeBinomial('D', mu=λ, alpha=φ, shape=λ.shape,
                                observed=df['deaths'])

        the_model = sts.ulam(data=df, cores=1)

    if verbose:
        sts.precis(the_model)

    return the_model


# Build all of the models and plot for comparison
interactions = ['A', 'P', 'A + P', 'F*A', 'F*P', 'F*A + F*P']
imodels = {k: make_interaction_model(k) for k in interactions}

models = [mGF] + list(imodels.values())
mnames = ['F'] + interactions

# Reset the model after plotting
pm.set_data(dict(F=df['femininity']), model=mGF.model)

ct = sts.coef_table(models, mnames, params=['β'])
cmp = sts.compare(models, mnames)

sts.plot_coef_table(ct, fignum=2)
sts.plot_compare(cmp['ct'], fignum=3)

# Best model: D ~ F + A + F*A

# TODO plot counterfactuals of best model vs baseline

# ----------------------------------------------------------------------------- 
#         12H4. Compare `damage_norm` and `log(damage_norm)`
# -----------------------------------------------------------------------------
df['logA'] = sts.standardize(np.log(df['damage_norm']))

with pm.Model():
    # Define the data
    F = pm.MutableData('F', df['femininity'])
    A = pm.MutableData('A', df['logA'])
    # Define the priors
    α = pm.Normal('α', 3, 0.5)
    β_F = pm.Normal('β_F', 0, 0.1)
    β_A = pm.Normal('β_A', 0, 0.1)
    β_FA = pm.Normal('β_FA', 0, 0.1)
    # Build the outcome variable
    λ = pm.Deterministic('λ', pm.math.exp(α + β_F*F + β_A*A + β_FA*F*A))
    φ = pm.Exponential('φ', 1)
    D = pm.NegativeBinomial('D', mu=λ, alpha=φ, shape=λ.shape,
                            observed=df['deaths'])
    mFA = sts.ulam(data=df, cores=1)

print('mFA:')
sts.precis(mFA)

models = [imodels['F*A'], mFA]
mnames = ['F*A', 'F*log(A)']

ct = sts.coef_table(models, mnames, params=['β'])
cmp = sts.compare(models, mnames)

sts.plot_compare(cmp['ct'], fignum=4)
sts.plot_coef_table(ct, fignum=5)

# Plot predictions at mean damage_norm
Fs = np.linspace(1, 11)
As = 0*Fs

fig, axs = plt.subplots(num=6, ncols=2, clear=True)
ax = axs[0]
ax.scatter('femininity', 'deaths', data=df, alpha=0.4)
ax.axhline(df['deaths'].mean(), ls='--', lw=1, c='gray', label='mean deaths')

# Plot posterior mean
λ_samp = sts.lmeval(mFA, out=mFA.model.λ, eval_at=dict(F=Fs, A=As))
D_samp = sts.lmeval(mFA, out=mFA.model.D, eval_at=dict(F=Fs, A=As))

# Plot posterior mean + predictive
sts.lmplot(mGF, fit_x=Fs, fit_y=λ_samp, ax=ax)
sts.lmplot(mGF, fit_x=Fs, fit_y=D_samp, line_kws=dict(c='none'), ax=ax)

ax.set(xlabel='femininity',
       ylabel='deaths',
       xticks=fem_levels,
       yscale='log')

# =============================================================================
# =============================================================================
