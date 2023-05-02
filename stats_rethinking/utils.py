#!/usr/bin/env python3
# =============================================================================
#     File: utils.py
#  Created: 2019-06-24 21:35
#   Author: Bernie Roesler
#
"""
  Description: Utility functions for Statistical Rethinking code.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import warnings

from scipy import stats, linalg
from scipy.interpolate import BSpline
from sklearn.utils.extmath import cartesian


def quantile(data, q=0.89, width=10, precision=8,
             q_func=np.quantile, verbose=False, **kwargs):
    """Pretty-print the desired quantile values from the data.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    width : int, optional, default=10
        Width of printing field.
    precision : int, optional, default=8
        Number of decimal places to print.
    q_func : callable, optional, default=numpy.quantile
        Function to compute the quantile outputs from the data.
    verbose : bool, optional, default=True
        Print the output quantile percentages names and values.
    **kwargs
        Additional arguments to `q_func`.

    Returns
    -------
    quantile : scalary or ndarray
        The requested quantiles. See documentation for `numpy.quantile`.

    See Also
    --------
    `numpy.quantile`
    """
    q = np.atleast_1d(q)
    quantiles = q_func(data, q, **kwargs)
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-1}g}%" for p in q])
        value_str = ' '.join([f"{q:{fstr}}" for q in quantiles])
        print(f"{name_str}\n{value_str}")
    return quantiles


def percentiles(data, q=50, **kwargs):
    r"""Pretty-print the desired percentile values from the data.

    .. note:: A wrapper around `quantile`, where the arguments are forced
        to take the form:
    .. math:: a = \frac{1 - q}{2}
        and called with :math:\mathtt{quantile(data, (a, 1-a))}

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100, inclusive.
    **kwargs
        See `quantile` for additional options.

    See Also
    --------
    quantile
    """
    a = (1 - (q/100)) / 2
    quantiles = quantile(data, (a, 1-a), **kwargs)
    return quantiles


# TODO remove width and precision arguments and just take fstr='8.2f', e.g.
def hpdi(data, alpha=None, q=None,
         verbose=False, width=10, precision=8, **kwargs):
    """Compute highest probability density interval.

    .. note::
        This function calls `sts.quantile` with `pymc.stats.hpd` function.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions
    alpha : array_like
    q : array_like
    verbose : bool
    width : int
    precision : int
    kwargs : dict_like

    Returns
    -------
    quantiles : (M, N) ndarray
        Matrix of M vectors in N dimensions

    Examples
    --------
    >>> arr = np.random.random((100, 1))
    >>> (sts.hpdi(arr, q=0.89) ==  sts.hpdi(arr, alpha=1-0.89)).all()
    True
    """
    if alpha is None:
        if q is None:
            alpha = 0.11
        else:
            alpha = 1 - q
    alpha = np.atleast_1d(alpha)
    q = 1 - alpha  # alpha takes precedence if both are given
    quantiles = np.array([az.hdi(np.asarray(data), hdi_prob=x, **kwargs).squeeze()
                          for x in q])
    if verbose:
        for i in range(len(alpha)):
            fstr = f"{width}.{precision}f"
            name_str = ' '.join([f"{100*x:{width-2}g}%" for x in np.hstack((q[i], q[i]))])
            value_str = ' '.join([f"{x:{fstr}}" for x in quantiles[i]])
            print(f"|{name_str}|\n{value_str}")
    return quantiles


def grid_binom_posterior(Np, k, n, prior_func=None, norm_post=True):
    """Posterior probability assuming a binomial distribution likelihood and
    arbitrary prior.

    Parameters
    ----------
    Np : int
        Number of parameter values to use.
    k : int
        Number of event occurrences observed.
    n : int
        Number of trials performed.
    prior_func : callable, optional, default U(0, 1)
        Function of one parameter describing the prior distribution.
        If prior_func is None, it defaults to a uniform prior
    norm_post : bool, optional, default True
        If True, normalize posterior to a maximum value of 1.

    Returns
    -------
    p_grid : (Np, 1) ndarray
        Vector of parameter values.
    posterior : (Np, 1) ndarray
        Vector of posterior probability values.
    """
    p_grid = np.linspace(0, 1, Np)  # vector of possible parameter values
    # default uniform prior
    prior = np.ones(Np) if prior_func is None else prior_func(p_grid)
    likelihood = stats.binom.pmf(k=k, n=n, p=p_grid)  # binomial distribution
    unstd_post = likelihood * prior                   # unstandardized posterior
    posterior = unstd_post / np.sum(unstd_post) if norm_post else unstd_post
    return p_grid, posterior, prior


def density(data, adjust=1.0, **kwargs):
    """Return the kernel density estimate of the data, consistent with
    R function of the same name.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in K dimensions.
    adjust : float, optional, default=1.0
        Multiplicative factor for the bandwidth.
    **kwargs : optional
        Additional arguments passed to `scipy.stats.gaussian_kde`.

    .. note:: This function overrides the `bw_method` argument. The
      stats_rethinking "dens" (R code 2.9) function calls the following
      R function:
          thed <- density(data, adjust=0.5)
      The default bandwidth in `density` (R docs) is: `bw="nrd0"`, which
      corresponds to 'silverman' in python. `adjust` sets `bandwith *= adjust`.

    Returns
    -------
    kde : kernel density estimate object
        Call kde.pdf(x) to get the actual samples

    """
    kde = stats.gaussian_kde(data, **kwargs)
    kde.set_bandwidth(adjust * kde.silverman_factor())
    return kde


# TODO expand documentation with examples
def expand_grid(**kwargs):
    """Return a DataFrame of points, where the columns are kwargs.

    Notes
    -----
    Compare to `numpy.meshgrid`[0]:
        xx, yy = np.meshgrid(mu_list, sigma_list)  # == (..., index='xy')
    `expand_grid` returns the *transpose* of meshgrid's default xy orientation.
    `expand_grid` matches:
        xx, yy = np.meshgrid(mu_list, sigma_list, index='ij')

    .. [0]: <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid>

    See Also
    --------
    numpy.meshgrid
    """
    return pd.DataFrame(cartesian(kwargs.values()), columns=kwargs.keys())


# TODO
#   * expand documentation with examples
#   * ignore unsupported columns like 'datetime' types
#   * remove dependence on input type. pd.DataFrame.from_dict? or kwarg?
#   * built-in verbose flag to print output with desired precision
def precis(quap, p=0.89):
    """Return a `DataFrame` of the mean, standard deviation, and percentile
    interval of the given `rv_frozen` distributions.

    Parameters
    ----------
    quap : array-like, DataFrame, or dict
        The model.
    p : float in [0, 1]
        The percentile of which to compute the interval.

    Returns
    -------
    result : DataFrame
        A DataFrame with a row for each variable, and columns for mean,
        standard deviation, and low/high percentiles of the variable.
    """
    a = (1-p)/2
    pp = 100*np.array([a, 1-a])  # percentages for printing

    # dictionary of `rv_frozen` distributions
    if isinstance(quap, dict):
        index = quap.keys()
        vals = np.empty((len(quap), 4))
        for i, v in enumerate(quap.values()):
            vals[i, :] = [v.mean(), v.std(), v.ppf(a), v.ppf(1-a)]
        df = pd.DataFrame(vals, index=index,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
                          )
        return df

    # DataFrame of data points
    if isinstance(quap, pd.DataFrame):
        index = quap.keys()
        df = pd.DataFrame()
        df['mean'] = quap.mean()
        df['std'] = quap.std()
        for i in range(2):
            df[f"{pp[i]:g}%"] = quap.apply(lambda x: np.nanpercentile(x, pp[i]))
        return df

    # Numpy array of data points
    if isinstance(quap, np.ndarray):
        # Columns are data, ignore index
        vals = np.vstack([np.nanmean(quap, axis=0),
                          np.nanstd(quap, axis=0),
                          np.nanpercentile(quap, pp[0], axis=0),
                          np.nanpercentile(quap, pp[1], axis=0)]).T
        df = pd.DataFrame(vals,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
                          )
        return df
    else:
        raise TypeError('quap of this type is unsupported!')


# TODO currently returns marginal posterior distributions of each variable, but
# could return joint distribution stats.multivariate_normal(mean, cov), where:
#   mean = np.r_[[map_est[v.name] for v in mvars]]
#   cov = linalg.inv(H)
def quap(vars=None, var_names=None, model=None, start=None):
    """Compute the quadratic approximation for the MAP estimate.

    Parameters
    ----------
    vars : list, optional, default=model.unobserved_RVs
        List of variables to optimize and set to optimum
    var_names : list, optional
        List of `str` of variables names specified by `model`
    model : pymc.Model (optional if in `with` context)
    start : `dict` of parameter values, optional, default=`model.test_point`

    Returns
    -------
    result : dict
        Dictionary of `scipy.stats.rv_frozen` distributions corresponding to
        the MAP estimates of `vars`.
    """
    if model is None:
        model = pm.modelcontext(model)

    pm.init_nuts(model=model)
    map_est = pm.find_MAP(start=start, model=model)

    if vars is None:
        if var_names is None:
            # filter out internally used variables
            mvars = [x for x in model.unobserved_RVs if not x.name.endswith('__')]
        else:
            mvars = [model[x] for x in var_names]
    else:
        mvars = vars

    # Need to compute *untransformed* Hessian! See ch02/quad_approx.py
    # See: <https://github.com/pymc-devs/pymc/issues/5443>
    for v in mvars:
        try:
            # Remove transform from the variable `v`
            model.rvs_to_transforms[v] = None
            # Change name so that we can use `map_est['v']` value
            v_value = model.rvs_to_values[v]
            v_value.name = v.name
        except KeyError:
            warnings.warn(f"Hessian for '{v.name}' may be incorrect!")
            continue

    # FIXME adding this as a dictionary item breaks precis (and probably other
    # functions) since the multivariate normal has different attributes from
    # the marginal distributions. Make `QuadApprox` class instead.
    # Entire posterior distribution estimate
    quap = dict()
    means = np.r_[[map_est[v.name] for v in mvars]]
    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    H = pm.find_hessian(map_est, vars=mvars, model=model)
    cov = linalg.inv(H)
    quap['posterior'] = stats.multivariate_normal(mean=means, cov=cov)

    # Create marginal distribution estimates
    stds = np.sqrt(np.diag(cov))

    for v, mean, std in zip(mvars, means, stds):
        if np.isnan(std) or (std < 0) or np.isnan(mean).any():
            raise ValueError(f"std('{v.name}') = {std} is invalid!"
                             + " Check testval of prior.")
        quap[v.name] = stats.norm(loc=mean, scale=std)

    return quap


# TODO
#   * make NormApprox class that contains the dictionary + method to get sizes
#     so we don't have to use `v.rvs().shape`
def sample_quap(quap, N=1000):
    """Sample each distribution in the `quap` dict.
    Return a dict like pm.sample_prior_predictive."""
    out = dict()
    for k, v in quap.items():
        # number of samples must be first dimension
        size = [N] + list(v.rvs().shape)
        # if len(size) == 1:
        #     size += [1]  # guarantee at least column vector
        out[k] = v.rvs(size=size)
    return out


def norm_fit(data, hist_kws=None, ax=None):
    """Plot a histogram and a normal curve fit to the data."""
    if ax is None:
        ax = plt.gca()
    if hist_kws is None:
        hist_kws = dict()
    sns.histplot(data, stat='density', alpha=0.4, ax=ax, **hist_kws)
    norm = stats.norm(data.mean(), data.std())
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 1000)
    y = norm.pdf(x)
    ax.plot(x, y, 'C0')
    return ax


def sample_to_dataframe(data):
    """Convert dict of samples to DataFrame."""
    try:
        df = pd.DataFrame(data)
    except:
        # if data has more than one dimension, enumerate the columns
        df = pd.DataFrame()
        for k, v in data.items():
            df_s = pd.DataFrame(v)

            # name the columns
            if v.ndim == 1:
                df_s.columns = [k]
            else:
                df_s = df_s.add_prefix(k + '__')  # enumerate matrix variables

            # concatenate into one DataFrame
            if df.empty:
                df = df_s
            else:
                df = df.join(df_s)
    return df


def standardize(x, data=None, axis=0):
    """Standardize the input vector `x` by the mean and std of `data`.

    .. note::
        The following lines are equivalent:
                           (x - x.mean()) / x.std() == stats.zscore(x, ddof=1)
        (N / (N-1))**0.5 * (x - x.mean()) / x.std() == stats.zscore(x, ddof=0)
        where N = x.size
    """
    if data is None:
        data = x
    return (x - data.mean(axis=axis)) / data.std(axis=axis)


def poly_weights(w, poly_order=0, include_const=True):
    """Return array of polynomial weight vectors."""
    w = np.asarray(w)

    if poly_order == 0:
        out = w
    else:
        try:
            out = np.vstack([w**(i+1) for i in range(poly_order)])
        except ValueError:
            raise ValueError(f"poly_order value '{poly_order}' is invalid")

    if include_const:
        out = np.vstack([np.ones_like(w), out])  # weight constant term == 1

    return out


def pad_knots(knots, k=3):
    """Repeat first and last knots `k` times."""
    knots = np.asarray(knots)
    return np.concatenate([np.repeat(knots[0], k),
                           knots,
                           np.repeat(knots[-1], k)])


def bspline_basis(x=None, t=None, k=3, padded_knots=False):
    """Create the B-spline basis matrix of coefficients.

    Parameters
    ----------
    x : array_like, optional
        points at which to evaluate the B-spline bases. If `x` is not given,
        a `scipy.interpolate.BSpline` object will be returned.
    t : array_like, shape (n+k+1,)
        internal knots
    k : int, optional, default=3
        B-spline order
    padded_knots : bool, optional, default=False
        if True, treat the input `t` as padded, otherwise, pad `t` with `k`
        each of the leading and trailing "border knot" values.

    Returns
    -------
    if `x` is given:
    B : ndarray, shape (x.shape, n+k+1)
        B-spline basis functions evaluated at the given points `x`. The last
        dimension is the number of knots.
    else:
    b : :obj:scipy.interpolate.BSpline
        B-spline basis function object with identity matrix as weights.
    """
    if t is None:
        raise TypeError("bspline_basis() missing 1 required keyword argument: 't'")
    if not padded_knots:
        t = pad_knots(t, k)
    m = len(t) - k - 1
    c = np.eye(m)  # only activate one basis at a time
    b = BSpline(t, c, k, extrapolate=False)
    if x is None:
        return b
    else:
        B = b(x)
        B[np.isnan(B)] = 0.0
        return B

# =============================================================================
# =============================================================================
