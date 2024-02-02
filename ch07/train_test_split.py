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
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from pathlib import Path
from tqdm.contrib.concurrent import process_map

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

DEBUG = True
FORCE_UPDATE = True  # if True, overwrite `tf_file` regardless

FLAT = 100  # `b_sigma` value for a "flat" prior.

# -----------------------------------------------------------------------------
#         Replicate the experiment Ne times for each k
# -----------------------------------------------------------------------------
if DEBUG:
    Ne = 10
    Ns = [20]
    params = np.arange(1, 6)
    b_sigmas = [FLAT]
else:
    Ne = 1000                       # replicates
    Ns = [20, 100]                  # data points
    params = np.arange(1, 6)        # parameters in the model
    b_sigmas = [FLAT, 1, 0.5, 0.2]  # regularization via small prior variance

tf_file = Path(f"./train_test_Ne{Ne:d}.pkl")

if not FORCE_UPDATE and tf_file.exists():
    tf = pd.read_pickle(tf_file)
else:
    # Parallelize All at Once:
    def exp_train_test(args):
        """Run a single simulation."""
        N, k, _, b = args
        ic = ((b == FLAT) or (b == 0.5))
        res = sts.sim_train_test(
            N,
            k,
            b_sigma=b,
            compute_WAIC=ic,
            compute_LOOIC=ic,
            compute_LOOCV=False,  # SLOW if True
        )
        return (res['res'], N, k, b)

    all_args = [
        (N, k, i, b)
        for N in Ns
        for k in params
        for i in range(Ne)
        for b in b_sigmas
    ]

    if DEBUG:
        # Non-parallel:
        from tqdm import tqdm
        res = [exp_train_test(x) for x in tqdm(all_args)]
    else:
        # Parallel:
        res = process_map(
            exp_train_test,
            all_args,
            chunksize=2*len(params),
            max_workers=8,
        )

    # Convert list of tuples (Series(), N, k, b) to DataFrame with
    # columns=Series.index.
    lres = list(zip(*res))
    tf = pd.DataFrame(lres[0])
    tf['N'] = lres[1]
    tf['params'] = lres[2]
    tf['b_sigma'] = lres[3]

    for c in ['WAIC', 'LOOIC', 'LOOCV']:
        if c in tf.columns:
            tf[(c, 'err')] = np.abs(tf[(c, 'test')] - tf[('deviance', 'test')])

    tf = tf.sort_index(axis='columns')

    # Save the data
    tf.to_pickle(tf_file)

# Compute the mean and std deviance for each number of parameters
df = tf.groupby(['b_sigma', 'N', 'params']).agg(['mean', 'std'])
df.columns.names = ['IC', 'kind', 'stat']

# -----------------------------------------------------------------------------
#         Plots
# -----------------------------------------------------------------------------
# Figure 7.7 -- deviance vs number of parameters
fig = plt.figure(1, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
jitter = 0.05  # separation between points in x-direction

pf = df['deviance']  # just plot the deviance

idx = pd.IndexSlice

for i, N in enumerate(Ns):
    ax = fig.add_subplot(gs[i])

    ix = idx[FLAT, N]
    ax.errorbar(params - jitter, pf.loc[ix, ('train', 'mean')],
                yerr=pf.loc[ix, ('train', 'std')],
                fmt='oC0', markerfacecolor='C0', ecolor='C0')
    ax.errorbar(params + jitter, pf.loc[ix, ('test', 'mean')],
                yerr=pf.loc[ix, ('test', 'std')],
                fmt='ok', markerfacecolor='none', ecolor='k')

    # Label the training set
    ax.text(
        x=(params - 4*jitter)[1],
        y=pf.loc[ix].iloc[1]['train', 'mean'],
        s='train',
        color='C0',
        ha='right',
        va='center',
    )

    # Label the test set
    ax.text(
        x=(params + 4*jitter)[1],
        y=pf.loc[ix].iloc[1]['test', 'mean'],
        s='test',
        color='k',
        ha='left',
        va='center',
    )

    ax.set_xticks(params, labels=params)
    ax.set(title=f"{N = } ({Ne} simulations)",
           xlabel='number of parameters',
           ylabel='deviance')


# Figure 7.9 -- deviance vs parameters and increasing regularization
fig = plt.figure(2, clear=True, constrained_layout=True)
fig.set_size_inches((10, 5), forward=True)
gs = fig.add_gridspec(nrows=1, ncols=2)

for i, N in enumerate(Ns):
    ax = fig.add_subplot(gs[i])

    ix = idx[FLAT, N]
    ax.scatter(params, pf.loc[ix, ('train', 'mean')], c='C0')
    ax.scatter(params, pf.loc[ix, ('test', 'mean')],
               facecolors='none', edgecolors='k',
               label=rf"$\mathcal{{N}}$(0, {FLAT})")

    # Only plot the regularized curves
    for b, ls in zip(b_sigmas[1:], ['--', ':', '-']):
        ix = idx[b, N]
        ax.plot(params, pf.loc[ix, ('train', 'mean')], c='C0', ls=ls)
        ax.plot(params, pf.loc[ix, ('test', 'mean')], c='k', ls=ls,
                label=rf"$\mathcal{{N}}$(0, {b})")

    ax.set_xticks(params, labels=params)
    ax.set(title=f"{N = } ({Ne} simulations)",
           xlabel='number of parameters',
           ylabel='deviance')

    if i == 0:
        # Add lines to the legend for train/test
        handles, labels = ax.get_legend_handles_labels()
        custom_lines = [Line2D([0], [0], color='C0', marker='o', lw=2),
                        Line2D([0], [0], color='k', marker='o',
                               markerfacecolor='none', lw=2)]
        handles.extend(custom_lines)
        labels.extend(['train', 'test'])
        ax.legend(handles, labels, loc='lower left')


# Figure 7.10 -- Test deviance/err with lines for each information criteria
def plot_ICs(N, kind, legend=False, ax=None):
    """Plot average deviance or error."""
    if ax is None:
        ax = plt.gca()

    # Plot one curve for flat priors, one for regularized
    bs, cs = ([FLAT], ['C0']) if DEBUG else ([FLAT, 0.5], ['C0', 'k'])
    for b, c in zip(bs, cs):
        ix = idx[b, N]

        # Points are mean test deviance, with different priors
        if kind == 'test':
            ax.scatter(params, df.loc[ix, ('deviance', 'test', 'mean')],
                       facecolors='none' if c == 'C0' else c, edgecolors=c)

        # Plot curves for each information criterion
        # for ic, ls in zip(['WAIC', 'LOOCV', 'LOOIC'], ['-', '--', '-.']):
        for ic, ls in zip(['WAIC', 'LOOIC'], ['-', '-.']):
            ax.plot(params, df.loc[ix, (ic, kind, 'mean')],
                    color=c, ls=ls, label=ic if c == 'C0' else None)

    ax.set_xticks(params, labels=params)
    ax.set(title=f"{N = } ({Ne} simulations)",
           xlabel='number of parameters',
           ylabel=('average deviance'
                   if kind == 'test'
                   else 'average error (test deviance)'))

    if legend and N == 20:
        # Add lines to the legend for scatter points + lines
        handles, labels = ax.get_legend_handles_labels()
        custom_lines = [Line2D([0], [0], color='C0', marker='o', lw=2),
                        Line2D([0], [0], color='k', marker='o',
                               markerfacecolor='none', lw=2)]
        handles.extend(custom_lines)
        labels.extend([f"σ = {FLAT}", 'σ = 0.5'])
        ax.legend(handles, labels)  # , loc='lower left')

    return ax


fig = plt.figure(3, clear=True, constrained_layout=True)
fig.set_size_inches((10, 10), forward=True)
gs = fig.add_gridspec(nrows=2, ncols=2)

for i, N in enumerate(Ns):
    plot_ICs(N, 'test', legend=True, ax=fig.add_subplot(gs[i, 0]))
    plot_ICs(N, 'err', legend=False, ax=fig.add_subplot(gs[i, 1]))


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
