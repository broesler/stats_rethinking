#!/usr/bin/env python3
# =============================================================================
#     File: 11H3.py
#  Created: 2024-01-11 17:51
#   Author: Bernie Roesler
#
r"""
Solution to 11H3. Bald Eagles data.

.. note:: The problem is incorrectly labeled in the digital book as 10H3.

The model will be:

.. math::
    \begin{align}
        y &\sim \mathrm{Binomial}(n, p) \\
        \log \frac{p}{1 - p} &= \alpha + \beta_P P + \beta_V V + \beta_A A \\
        \alpha \sim \mathcal{N}(0, 10) \\
        \beta_P \sim \mathcal{N}(0, 5) \\
        \beta_V \sim \mathcal{N}(0, 5) \\
        \beta_A \sim \mathcal{N}(0, 5) \\
    \end{align}

where
:math:`y` is the number of successful attempts,
:math:`n` is the total number of attemps,
:math:`P` indicates whether the pirate had a large body size,
:math:`V` indicates whether the victim had a large body size, and
:math:`A` indicates whether the pirate was an adult.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(Path('../data/eagles.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 8 entries, 0 to 7
# Data columns (total 5 columns):
#    Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   y       8 non-null      int64    # number of successful attempts
#  1   n       8 non-null      int64    # total number of attempts
#  2   P       8 non-null      object   # whether the pirate had a large body
#  3   A       8 non-null      object   # whether the victim had a large body
#  4   V       8 non-null      object   # whether the pirate was an adult
# dtypes: int64(2), object(3)
# memory usage: 452.0 bytes

# -----------------------------------------------------------------------------
#         (a) Build the model and compare quap and ulam
# -----------------------------------------------------------------------------
with pm.Model() as the_model:
    P = df['P'] == 'L'
    V = df['V'] == 'L'
    A = df['A'] == 'A'
    α = pm.Normal('α', 0, 10)
    β_P = pm.Normal('β_P', 0, 5)
    β_V = pm.Normal('β_V', 0, 5)
    β_A = pm.Normal('β_A', 0, 5)
    p = pm.Deterministic('p', pm.math.invlogit(α + β_P*P + β_V*V + β_A*A))
    y = pm.Binomial('y', df['n'], p, observed=df['y'])
    m_quap = sts.quap(data=df)
    m_ulam = sts.ulam(data=df)

print('quap:')
sts.precis(m_quap)
print('ulam:')
sts.precis(m_ulam)

models = [m_quap, m_ulam]
mnames = ['quap', 'ulam']
ct = sts.coef_table(models, mnames)
sts.plot_coef_table(ct, fignum=1)

# TODO plot sample distributions of each parameter for each model?

# -----------------------------------------------------------------------------
#         (b) Plot the posterior predictions
# -----------------------------------------------------------------------------
# (1) The predicted probability of success is just the mean of `p`.
post_p = m_ulam.deterministics['p']
print('p:')
sts.precis(post_p)

post_y = (
    pm.sample_posterior_predictive(
        trace=m_ulam.get_samples(),
        model=m_ulam.model
    )
    .posterior_predictive['y']
)
print('y:')
sts.precis(post_y)

# Plot posterior predictions using postcheck
ax = sts.postcheck(
    m_ulam,
    mean_name='p',
    mean_transform=lambda x: x * df['n'].values,
    fignum=2
)
ax.set(title='posterior predictions (natural scale)')

# Plot posterior predictions using postcheck
ax = sts.postcheck(
    m_ulam,
    mean_name='p',
    agg_name='n',
    fignum=3
)
ax.set_ylabel('p')

ax.set(title='posterior predictions (probabilty scale)')

# -----------------------------------------------------------------------------
#         (c) Add an interaction to the model
# -----------------------------------------------------------------------------
with pm.Model():
    P = df['P'] == 'L'
    V = df['V'] == 'L'
    A = df['A'] == 'A'
    α = pm.Normal('α', 0, 10)
    β_P = pm.Normal('β_P', 0, 5)
    β_V = pm.Normal('β_V', 0, 5)
    β_A = pm.Normal('β_A', 0, 5)
    β_PA = pm.Normal('β_PA', 0, 5)
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(α + β_P*P + β_V*V + β_A*A + β_PA*A*P)
    )
    y = pm.Binomial('y', df['n'], p, observed=df['y'])
    m_int = sts.ulam(data=df)

print('Interaction:')
sts.precis(m_int)

models = [m_ulam, m_int]
mnames = ['simple', 'interaction'] 

sts.plot_coef_table(sts.coef_table(models, mnames), fignum=4)

cmp = sts.compare(models, mnames)['ct']
print(cmp)
# Interaction term wins!

# =============================================================================
# =============================================================================
