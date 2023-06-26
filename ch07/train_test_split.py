#!/usr/bin/env python3
# =============================================================================
#     File: train_test_split.py
#  Created: 2023-06-21 16:46
#   Author: Bernie Roesler
#
"""
§7.2--7.4 Train/Test Split and Akaike Information Criterion
"""
# =============================================================================

import matplotlib.pyplot as plt
# import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# (R code 7.17 - 7.19)
Y_SIGMA = 1


# -----------------------------------------------------------------------------
#         Define functions
# -----------------------------------------------------------------------------
def lppd(m, data_in, data_out, Ns=1000):
    """Compute the log pointwise predictive density for a model."""
    mu_samp = sts.lmeval(m, out=m.model.μ, eval_at={'X': data_in}, N=Ns)
    y_logp = stats.norm(mu_samp, Y_SIGMA).logpdf(np.c_[data_out])
    return sts.log_sum_exp(y_logp, axis=1) - np.log(Ns)


def sim_train_test(N=20, k=3, rho=np.r_[0.15, -0.4], b_sigma=100):
    """Simulate fitting a model of `k` parameters to `N` data points.

    Parameters
    ----------
    N : int
        The number of simulated data points.
    k : int
        The number of parameters in the linear model, including the intercept.
    rho : 1-D array_like
        A vector of "true" parameter values, excluding the intercept term.
    b_sigma : float
        The model standard deviation of the slope terms.

    Returns
    -------
    result : dict
        'dev' : dict with keys {'train', 'test'}
            The deviance of the model for the train and test data.
        'model' : :obj:Quap
            The model itself.
    """

    # Define the dimensions of the "true" distribution to match the model
    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k

    # Generate train/test data
    # NOTE this method of creating the data obfuscates the underlying linear
    # model:
    #   y_i ~ N(μ_i, 1)
    #   μ_i = α + β_1 * x_{1,i} + β_2 * x_{2,i}
    #   α = 0
    #   β_1 =  0.15
    #   β_2 = -0.4
    Rho = np.eye(n_dim)
    Rho[0, 1:len(rho)+1] = rho
    Rho[1:len(rho)+1, 0] = rho
    # >>> Rho
    # === array([[ 1.  ,  0.15, -0.4 ],
    #            [ 0.15,  1.  ,  0.  ],
    #            [-0.4 ,  0.  ,  1.  ]])
    #

    true_dist = stats.multivariate_normal(mean=np.zeros(n_dim), cov=Rho)
    X_train = true_dist.rvs(N)  # (N, k)
    X_test = true_dist.rvs(N)

    # Define the training matrix
    mm_train = np.ones((N, 1))  # intercept term
    if k > 1:
        mm_train = np.c_[mm_train, X_train[:, 1:k]]

    # Build and fit the model to the training data
    with pm.Model():
        X = pm.MutableData('X', mm_train)
        if k == 1:
            α = pm.Normal('α', 0, b_sigma)
            μ = pm.Deterministic('μ', α)
            y = pm.Normal('y', μ, Y_SIGMA, observed=X_train[:, 0])
        else:
            α = pm.Normal('α', 0, b_sigma, shape=(1,))
            βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
            β = pm.math.concatenate([α, βn])
            μ = pm.Deterministic('μ', pm.math.dot(X, β))
            y = pm.Normal('y', μ, Y_SIGMA,
                          observed=X_train[:, 0],
                          shape=X[:, 0].shape
                          )
        quap = sts.quap()

    # Compute the deviance
    dev = pd.Series(index=['train', 'test'], dtype=float)
    dev['train'] = -2 * np.sum(
            lppd(quap, data_in=mm_train, data_out=X_train[:, 0])
        )

    # Compute the lppd with the test data
    mm_test = np.ones((N, 1))
    if k > 1:
        mm_test = np.c_[mm_test, X_test[:, 1:k]]

    dev['test'] = -2 * np.sum(
            lppd(quap, data_in=mm_test, data_out=X_test[:, 0])
        )

    return dict(dev=dev, model=quap)


# -----------------------------------------------------------------------------
#         Replicate the experiment Ne times for each k
# -----------------------------------------------------------------------------
Ne = 100                  # number of replicates
Ns = [20, 100]            # data points
params = np.arange(1, 6)  # number of parameters in the model


FORCE_UPDATE = True
tf_file = Path('./train_test_all.pkl')

if not FORCE_UPDATE and tf_file.exists():
    tf = pd.read_pickle(tf_file)
else:
    # Parallelize All at Once:
    def exp_train_test(args):
        """Run a single simulation."""
        N, k, _ = args
        return (sim_train_test(N, k)['dev'], N, k)

    all_args = [
            (N, k, i)
            for N in Ns
            for k in params
            for i in range(Ne)
        ]
    res = process_map(
            exp_train_test,
            all_args,
            max_workers=16,
            chunksize=2*len(params)
        )

    # Convert list of tuples (Series(), N, k) to DataFrame with
    # columns=Series.index.
    lres = list(zip(*res))
    tf = pd.concat(lres[0], axis='columns').T
    tf['N'] = lres[1]
    tf['params'] = lres[2]

    # Save the data
    tf.to_pickle(tf_file)

# Compute the mean and std deviance for each number of parameters
df = tf.groupby(['N', 'params']).agg(['mean', 'std'])
df.columns.names = ['kind', 'stat']

# Figure 7.7
fig = plt.figure(1, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
jitter = 0.05  # separation between points in x-direction

for i, N in enumerate(Ns):
    ax = fig.add_subplot(gs[i])

    ax.errorbar(params-jitter, df.loc[N, ('train', 'mean')],
                yerr=df.loc[N, ('train', 'std')],
                fmt='oC0', markerfacecolor='C0', ecolor='C0')
    ax.errorbar(params+jitter, df.loc[N, ('test', 'mean')],
                yerr=df.loc[N, ('test', 'std')],
                fmt='ok', markerfacecolor='none', ecolor='k')

    # Label the training set
    ax.text(x=(params - 4*jitter)[1],
            y=df.loc[N].iloc[1]['train', 'mean'],
            s='train',
            color='C0',
            ha='right',
            va='center',
            )

    # Label the test set
    ax.text(x=(params + 4*jitter)[1],
            y=df.loc[N].iloc[1]['test', 'mean'],
            s='test',
            color='k',
            ha='left',
            va='center',
            )

    ax.set_xticks(params, labels=params)
    ax.set(title=f"{N = }",
           xlabel='number of parameters',
           ylabel='deviance')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
