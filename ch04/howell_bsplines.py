#!/usr/bin/env python3
#==============================================================================
#     File: howell_bsplines.py
#  Created: 2019-08-03 08:48
#   Author: Bernie Roesler
#
"""
  Description: Howell model using B-splines (Section 4.5.2)
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

from scipy import stats
from scipy.interpolate import BSpline, splev
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Load Dataset
#------------------------------------------------------------------------------
data_path = '../data/'

# df: year, doy (day # of year), temp [C], temp_upper [C], temp_lower [C]
df = pd.read_csv(data_path + 'cherry_blossoms.csv')

print(sts.precis(df))

# Parse year + day of year into an actual datetime column
def convert_time(row):
    """Convert row with 'year' and 'doy' (day of year) columns to datetime."""
    return pd.datetime(int(row['year']), 1, 1) \
            + pd.Timedelta(days=int(row['doy'] - 1))

df['datetime'] = df.loc[~df['doy'].isnull()].apply(convert_time, axis=1)

# Ns = 10_000  # general number of samples to use

# Plot the temperature vs time
fig = plt.figure(1, clear=True, figsize=(12, 4))
ax = fig.add_subplot()
ax.scatter(df['year'], df['temp'], alpha=0.5, label='Raw Data')
ax.set(xlabel='Year',
       ylabel='Temperature [°C]')
ax.legend()
plt.tight_layout()

#------------------------------------------------------------------------------ 
#        Build B-Splines (R code 4.73 - 4.75
#------------------------------------------------------------------------------
df = df.dropna(subset=['temp'])  # only keep rows with temperature data

# Choose number of knots 
Nk = 15

# Evenly space knots along percentiles of input variable
# df['year'] is ~uniform so knots will be as well
knots = sts.quantile(df['year'], q=np.linspace(0, 1, Nk))

# Test we've covered the full range of dates
assert knots[0] == df['year'].min()
assert knots[-1] == df['year'].max()

# Build the basis functions
k = 3  # degree of spline (3 == cubic)

def pad_knots(knots, k=3):
    """Repeat first and last knots `k` times."""
    knots = np.asarray(knots)
    return np.concatenate([np.repeat(knots[0], k),
                           knots,
                           np.repeat(knots[-1], k)])


def bspline_basis(x, t, k=3, padded_knots=False):
    """Create the B-spline basis matrix of coefficients.

    Parameters
    ----------
    x : array_like
        points at which to evaluate the B-spline bases
    t : array_like, shape (n+k+1,)
        internal knots
    k : int, optional, default=3
        B-spline order
    padded_knots : bool, optional, default=False
        if True, treat the input `t` as padded, otherwise, pad `t` with `k`
        each of the leading and trailing "border knot" values.

    Returns
    -------
    B : ndarray, shape (x.shape, n+k+1)
        B-spline basis functions evaluated at the given points `x`. The last
        dimension is the number of knots.
    """
    if not padded_knots:
        t = pad_knots(t, k)
    m = len(t) - k - 1
    c = np.eye(m)  # only activate one basis at a time
    b = BSpline(t, c, k, extrapolate=False)
    B = b(x)
    B[np.isnan(B)] = 0.0
    return B

x = df['year']
B = bspline_basis(x, t=knots, k=k)

fig = plt.figure(2, clear=True)
ax = fig.add_subplot()

# Plot each basis function
for i in range(B.shape[1]):
    ax.plot(x, B[:, i], 'k-', alpha=0.5)

# Mark the knots
ax.plot(knots, 1.05*np.ones_like(knots), 'k+', markersize=10, alpha=0.5)

ax.set(xlabel=x.name,
       ylabel='Basis Value')
plt.tight_layout()


#==============================================================================
#==============================================================================
