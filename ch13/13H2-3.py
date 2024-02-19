#!/usr/bin/env python3
# =============================================================================
#     File: 13H2-3.py
#  Created: 2024-02-19 10:25
#   Author: Bernie Roesler
#
"""
13H2-3. The Trolley Problem, with varying intercepts.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(
    Path('../data/Trolley.csv'),
    dtype=dict(
        id='category',
        story='category',
    )
)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 9930 entries, 0 to 9929
# Data columns (total 12 columns):
#    Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   case       9930 non-null   object
#  1   response   9930 non-null   int64
#  2   order      9930 non-null   int64
#  3   id         9930 non-null   object
#  4   age        9930 non-null   int64
#  5   male       9930 non-null   int64
#  6   edu        9930 non-null   object
#  7   action     9930 non-null   int64
#  8   intention  9930 non-null   int64
#  9   contact    9930 non-null   int64
#  10  story      9930 non-null   object
#  11  action2    9930 non-null   int64
# dtypes: int64(8), object(4)
# memory usage: 931.1 KB

K = len(df['response'].unique())  # number of responses
Km1 = K - 1

Nu = len(df['id'].cat.categories)     # number of unique participants
Ns = len(df['story'].cat.categories)  # number of unique stories

# -----------------------------------------------------------------------------
#         Explore the data
# -----------------------------------------------------------------------------
a = (1 - 0.50) / 2  # only report 1 standard error

g_s = df.groupby('story')['response']
s_mean = g_s.mean()
s_ci = g_s.quantile([a, 1-a]).unstack('story')
s_err = np.abs(s_ci - s_mean)

N_u = 30  # only show 30 participants for brevity
g_u = df.groupby('id')['response']
u_mean = g_u.mean().iloc[:N_u]
u_ci = g_u.quantile([a, 1-a])[u_mean.index].unstack('id')
u_err = np.abs(u_ci - u_mean)

fig, axs = plt.subplots(num=1, ncols=2, sharey=True, clear=True)
fig.set_size_inches((10, 4), forward=True)
ax = axs[0]
ax.errorbar(s_mean.index.codes, s_mean, yerr=s_err,
            c='C3', ls='none', marker='o')
ax.set(xlabel='Story', ylabel='Response')
ax.set_xticks(s_mean.index.codes, labels=s_mean.index)

ax = axs[1]
ax.errorbar(u_mean.index.codes, u_mean, yerr=u_err,
            c='C3', ls='none', marker='o')
ax.set(xlabel='Particpant')

# -----------------------------------------------------------------------------
#         Build a linear model (no interaction)
# -----------------------------------------------------------------------------
# See Lecture 11 @ 34:38
# <https://youtu.be/VVQaIkom5D0?si=HiTUHROTeHSxj14L&t=2078>
with pm.Model() as model:
    A = pm.MutableData('A', df['action'])
    I = pm.MutableData('I', df['intention'])
    C = pm.MutableData('C', df['contact'])
    # Slope priors
    β_A = pm.Normal('β_A', 0, 0.5)
    β_I = pm.Normal('β_I', 0, 0.5)
    β_C = pm.Normal('β_C', 0, 0.5)
    # Intercept prior
    cutpoints = pm.Normal('cutpoints', 0, 1, shape=(Km1,),
                          transform=pm.distributions.transforms.ordered,
                          initval=np.linspace(-2, 3, Km1))
    # The linear model
    φ = pm.Deterministic('φ', β_A*A + β_C*C + β_I*I)
    # The response
    R = pm.OrderedLogistic('R', cutpoints=cutpoints, eta=φ, shape=φ.shape,
                           observed=df['response'].cat.codes)
    m_RX = sts.ulam(data=df, nuts_sampler='numpyro')

print('m_RX:')
sts.precis(m_RX)

# =============================================================================
# =============================================================================
