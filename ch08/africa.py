#!/usr/bin/env python3
# =============================================================================
#     File: africa.py
#  Created: 2023-09-18 17:59
#   Author: Bernie Roesler
#
"""
§8.1 Conditional modeling of terrain ruggedness vs GDP.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

# plt.style.use('seaborn-v0_8-darkgrid')

# Get the data (R code 8.1)
df = pd.read_csv(Path('../data/rugged.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 234 entries, 0 to 233
# Data columns (total 51 columns):
#    Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   isocode                 234 non-null    object
#  1   isonum                  234 non-null    int64
#  2   country                 234 non-null    object
#  3   rugged                  234 non-null    float64
#  4   rugged_popw             234 non-null    float64
#  5   rugged_slope            234 non-null    float64
#  6   rugged_lsd              234 non-null    float64
#  7   rugged_pc               234 non-null    float64
#  8   land_area               230 non-null    float64
#  9   lat                     234 non-null    float64
#  10  lon                     234 non-null    float64
#  11  soil                    225 non-null    float64
#  12  desert                  234 non-null    float64
#  13  tropical                234 non-null    float64
#  14  dist_coast              234 non-null    float64
#  15  near_coast              234 non-null    float64
#  16  gemstones               234 non-null    int64
#  17  rgdppc_2000             170 non-null    float64
#  18  rgdppc_1950_m           137 non-null    float64
#  19  rgdppc_1975_m           137 non-null    float64
#  20  rgdppc_2000_m           159 non-null    float64
#  21  rgdppc_1950_2000_m      137 non-null    float64
#  22  q_rule_law              197 non-null    float64
#  23  cont_africa             234 non-null    int64
#  24  cont_asia               234 non-null    int64
#  25  cont_europe             234 non-null    int64
#  26  cont_oceania            234 non-null    int64
#  27  cont_north_america      234 non-null    int64
#  28  cont_south_america      234 non-null    int64
#  29  legor_gbr               211 non-null    float64
#  30  legor_fra               211 non-null    float64
#  31  legor_soc               211 non-null    float64
#  32  legor_deu               211 non-null    float64
#  33  legor_sca               211 non-null    float64
#  34  colony_esp              234 non-null    int64
#  35  colony_gbr              234 non-null    int64
#  36  colony_fra              234 non-null    int64
#  37  colony_prt              234 non-null    int64
#  38  colony_oeu              234 non-null    int64
#  39  africa_region_n         234 non-null    int64
#  40  africa_region_s         234 non-null    int64
#  41  africa_region_w         234 non-null    int64
#  42  africa_region_e         234 non-null    int64
#  43  africa_region_c         234 non-null    int64
#  44  slave_exports           234 non-null    float64
#  45  dist_slavemkt_atlantic  57 non-null     float64
#  46  dist_slavemkt_indian    57 non-null     float64
#  47  dist_slavemkt_saharan   57 non-null     float64
#  48  dist_slavemkt_redsea    57 non-null     float64
#  49  pop_1400                201 non-null    float64
#  50  european_descent        165 non-null    float64
# dtypes: float64(31), int64(18), object(2)
# memory usage: 93.4 KB

# Log version of the output
df['log_GDP'] = np.log(df['rgdppc_2000'])

# Extract countries with GDP data
df = df.dropna(subset='rgdppc_2000')

# Rescale variables
df['log_GDP_std'] = df['log_GDP'] / df['log_GDP'].mean()  # proportion of avg
df['rugged_std'] = df['rugged'] / df['rugged'].max()      # [0, 1]

# Split into African and non-African countries
dA1 = df.loc[df['cont_africa'] == 1]
dA0 = df.loc[df['cont_africa'] == 0]


# -----------------------------------------------------------------------------
#         Model African and non-African countries separately
# -----------------------------------------------------------------------------
# (R code 8.2)
def gdp_model(x='rugged_std', y='log_GDP_std', data=df, a_std=1, b_std=1):
    """Create a model of log GDP vs ruggedness of terrain."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        obs = pm.MutableData('obs', data[y])
        a = pm.Normal('a', 1, a_std)
        b = pm.Normal('b', 0, b_std)
        μ = pm.Deterministic('μ', a + b * (ind - data[x].mean()))
        σ = pm.Exponential('σ', 1)
        y = pm.Normal('y', μ, σ, observed=obs, shape=ind.shape)
        return sts.quap(data=data)


# Model of African countries
m8_1_weak = gdp_model(data=dA1)

# Plot the prior predictive (R code 8.3)
N_lines = 50
rs = np.r_[-0.1, 1.1]
with m8_1_weak.model:
    pm.set_data({'ind': rs})
    idata_w = pm.sample_prior_predictive(N_lines)

fig = plt.figure(1, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(ncols=2)
ax = fig.add_subplot(gs[0])

# Plot poor priors
ax.axhline(df['log_GDP_std'].min(), c='k', ls='--', lw=1)
ax.axhline(df['log_GDP_std'].max(), c='k', ls='--', lw=1)

ax.plot(rs, idata_w.prior['μ'].mean('chain').T, c='k', alpha=0.3)

ax.set(title=(r'$a \sim \mathcal{N}(1, 1)$'
              '\n'
              r'$b \sim \mathcal{N}(0, 1)$'),
       xlabel='ruggedness',
       ylabel='log GDP (prop. of mean)',
       xlim=(0, 1),
       ylim=(0.5, 1.5))

# Print proportion of slopes > 0.6 (maximum reasonable) (R code 8.4)
print(
    "Slopes > 0.6: "
    f"{float(np.sum(np.abs(idata_w.prior['b']) > 0.6) / idata_w.prior['b'].size)}"
)

# Plot better priors (R code 8.5)
m8_1 = gdp_model(data=dA1, a_std=0.1, b_std=0.3)
with m8_1.model:
    pm.set_data({'ind': rs})
    idata = pm.sample_prior_predictive(N_lines)

ax = fig.add_subplot(gs[1], sharex=ax, sharey=ax)

ax.axhline(df['log_GDP_std'].min(), c='k', ls='--', lw=1)
ax.axhline(df['log_GDP_std'].max(), c='k', ls='--', lw=1)

ax.plot(rs, idata.prior['μ'].mean('chain').T, c='k', alpha=0.3)

ax.set(title=(r'$a \sim \mathcal{N}(1, 0.1)$'
              '\n'
              r'$b \sim \mathcal{N}(0, 0.3)$'),
       xlabel='ruggedness')
ax.tick_params(axis='y', left=False, labelleft=False)

# Non-African nations (R code 8.6)
m8_2 = gdp_model(data=dA0, a_std=0.1, b_std=0.25)

# See difference
print('m8.1:')
sts.precis(m8_1)
#     mean    std   5.5%  94.5%
# a 0.8817 0.0148 0.8580 0.9053
# b 0.1309 0.0713 0.0170 0.2448     ** positive slope
# σ 0.1048 0.0106 0.0879 0.1217

print('m8.2:')
sts.precis(m8_2)
#      mean    std    5.5%   94.5%
# a  1.0485 0.0101  1.0324  1.0646
# b -0.1405 0.0552 -0.2287 -0.0523  ** negative slope
# σ  0.1113 0.0071  0.0999  0.1227

# -----------------------------------------------------------------------------
#         Model the entire dataset (R code 8.7)
# -----------------------------------------------------------------------------
m8_3 = gdp_model(data=df, a_std=0.1, b_std=0.3)

print('m8.3:')
sts.precis(m8_3)

#     mean    std    5.5%  94.5%
# a 1.0000 0.0104  0.9834 1.0166
# b 0.0020 0.0548 -0.0856 0.0896
# σ 0.1365 0.0074  0.1247 0.1483

# Make another model with an indicator variable for the intercept (R code 8.8)
df['cid'] = (df['cont_africa'] == 1).astype(int)

with pm.Model():
    x, y = 'rugged_std', 'log_GDP_std'
    a_std, b_std = 0.1, 0.3
    ind = pm.MutableData('ind', df[x])
    obs = pm.MutableData('obs', df[y])
    cid = pm.MutableData('cid', df['cid'])
    a = pm.Normal('a', 1, a_std, shape=(2,))
    b = pm.Normal('b', 0, b_std)
    μ = pm.Deterministic('μ', a[cid] + b * (ind - df[x].mean()))
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=obs, shape=ind.shape)
    m8_4 = sts.quap(data=df)

# (R code 8.10)
ct = sts.compare(models=[m8_3, m8_4], mnames=['m8.3', 'm8.4'])['ct']
print('m8.3 vs m8.4:')
print(ct.xs('y'))
#          WAIC     SE  dWAIC    dSE  penalty    weight
# model
# m8.3  -188.93  13.24  62.77  14.99     2.60  2.34e-14
# m8.4  -251.70  15.48   0.00    NaN     4.56  1.00e+00

# (R code 8.11)
print('m8.4:')
sts.precis(m8_4)

#         mean    std    5.5%  94.5%
# a__0  1.0492 0.0102  1.0329 1.0654  # not Africa
# a__1  0.8804 0.0159  0.8549 0.9059  # Africa
# b    -0.0465 0.0457 -0.1195 0.0265
# σ     0.1124 0.0061  0.1027 0.1221

# Plot posterior predictions (R code 8.12)
rugged_seq = np.linspace(-0.1, 1.1, 30)

fig, ax = plt.subplots(num=2, clear=True, constrained_layout=True)

# Plot *not* Africa
for is_Africa in [0, 1]:
    if is_Africa:
        cid = np.ones_like(rugged_seq).astype(int)
        c = fc = 'C0'
        label = 'Africa'
    else:
        cid = np.zeros_like(rugged_seq).astype(int)
        c = 'k'
        fc = 'none'
        label = 'Not Africa'
    # NOTE explicitly evaluate the model, because lmplot expects the `x` kwarg
    # to be the same as the name of the independent variable in the model. In
    # this case, 'ind' ≠ 'rugged_std', so we get an error.
    mu_samp = sts.lmeval(
        m8_4,
        out=m8_4.model.μ,
        eval_at={'ind': rugged_seq, 'cid': cid},
    )
    sts.lmplot(
        fit_x=rugged_seq, fit_y=mu_samp,
        x='rugged_std', y='log_GDP_std', data=df.loc[df['cid'] == is_Africa],
        q=0.97,
        line_kws=dict(c=c),
        fill_kws=dict(facecolor=c),
        marker_kws=dict(edgecolor=c, facecolor=fc, lw=2),
        label=label,
        ax=ax,
    )
    ax.text(
        x=0.8, 
        y=1.02*mu_samp.mean('draw')[int(0.8*len(mu_samp))],
        s=label,
        c=c
    )

ax.spines[['right', 'top']].set_visible(False)
ax.set(title='m8.4',
       xlabel='ruggedness',
       ylabel='log GDP (prop. of mean)')


plt.ion()
plt.show()
# =============================================================================
# =============================================================================
