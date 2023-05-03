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


def percentiles(data, q=0.89, **kwargs):
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
        0 and 1, inclusive.
    **kwargs
        See `quantile` for additional options.

    See Also
    --------
    quantile
    """
    a = (1 - (q)) / 2
    quantiles = quantile(data, (a, 1-a), **kwargs)
    return quantiles


# TODO remove width and precision arguments and just take fstr='8.2f', e.g.
# * add axis=1 argument
# * allow multiple qs, but print them "nested" like on an x-axis.
def hpdi(data, q=0.89, verbose=False, width=6, precision=4, **kwargs):
    """Compute highest probability density interval.

    .. note::
        This function calls `sts.quantile` with `pymc.stats.hpd` function.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions
    q : array_like
    verbose : bool
    width : int
    precision : int
    kwargs : dict_like

    Returns
    -------
    quantiles : (M, N) ndarray
        Matrix of M vectors in N dimensions
    """
    q = np.atleast_1d(q)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        quantiles = np.array([az.hdi(np.asarray(data), hdi_prob=x, **kwargs)
                                .squeeze()
                            for x in q]).squeeze()
    # need at least 2 dimensions for printing
    # if quantiles.ndim >= 3:
    #     quantiles = quantiles.squeeze()
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*x:{width-2}g}%" for x in np.r_[q, q]])
        value_str = ' '.join([f"{x:{fstr}}" for x in quantiles])
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
    unstd_post = likelihood * prior                   # unnormalized posterior
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
    Compare to `numpy.meshgrid`:
        xx, yy = np.meshgrid(mu_list, sigma_list)  # == (..., index='xy')
    `expand_grid` returns the *transpose* of meshgrid's default xy orientation.
    `expand_grid` matches:
        xx, yy = np.meshgrid(mu_list, sigma_list, index='ij')

    See Also
    --------
    numpy.meshgrid
    """
    return pd.DataFrame(cartesian(kwargs.values()), columns=kwargs.keys())


# TODO
#   * expand documentation with examples
#   * ignore unsupported columns like 'datetime' types
#   * remove dependence on input type. pd.DataFrame.from_dict? or kwarg?
#       R version uses a LOT of "setMethod" calls to allow function to work
#       with many different datatypes.
#       See: <https://github.com/rmcelreath/rethinking/blob/master/R/precis.r>
#   * built-in verbose flag to print output with desired precision
def precis(obj, p=0.89, digits=4, verbose=True):
    """Return a `DataFrame` of the mean, standard deviation, and percentile
    interval of the given `rv_frozen` distributions.

    Parameters
    ----------
    quap : array-like, DataFrame, or dict
        The model.
    p : float in [0, 1]
        The percentile of which to compute the interval.
    digits : int
        Number of digits in the printed output if `verbose=True`.
    verbose : bool
        If True, print the output.

    Returns
    -------
    result : DataFrame
        A DataFrame with a row for each variable, and columns for mean,
        standard deviation, and low/high percentiles of the variable.
    """
    if not isinstance(obj, (Quap, pd.DataFrame, np.ndarray)):
        raise TypeError(f"quap of type '{type(quap)}' is unsupported!")

    a = (1-p)/2
    pp = 100*np.array([a, 1-a])  # percentages for printing

    if isinstance(obj, Quap):
        # Compute density intervals
        z = stats.norm.ppf(1 - a)
        lo = obj.coef - z * obj.std
        hi = obj.coef + z * obj.std
        df = pd.concat([obj.coef, obj.std, lo, hi], axis=1)
        df.columns = ['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]

    # DataFrame of data points
    if isinstance(obj, pd.DataFrame):
        df = pd.DataFrame()
        df['mean'] = obj.mean()
        df['std'] = obj.std()
        for i in range(2):
            df[f"{pp[i]:g}%"] = obj.apply(lambda x: np.nanpercentile(x, pp[i]))

    # Numpy array of data points
    if isinstance(obj, np.ndarray):
        # Columns are data, ignore index
        vals = np.vstack([np.nanmean(obj, axis=0),
                          np.nanstd(obj, axis=0),
                          np.nanpercentile(obj, pp[0], axis=0),
                          np.nanpercentile(obj, pp[1], axis=0)]).T
        df = pd.DataFrame(vals,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
                          )

    if verbose:
        with pd.option_context('display.float_format',
                               f"{{:.{digits}f}}".format):
            print(df)

    return df


# TODO `pairs` method to plot pair-wise covariance
class Quap():
    """The quadratic (*i.e.* Gaussian) approximation of the posterior.

    Attributes
    ----------
    coef : dict
        Dictionary of maximum *a posteriori* (MAP) coefficient values.
    cov : (M, M) DataFrame
        Covariance matrix.
    data : (M, N) array_like
        Matrix of the data used to compute the likelihood.
    map_est : dict
        Maximum *a posteriori* estimates of any Deterministic or Potential
        variables.
    model : :obj:`pymc.Model`
        The pymc model object used to define the posterior.
    start : dict
        Initial parameter values for the MAP optimization. Defaults to
        `model.initial_point`.
    """
    def __init__(self, coef=None, cov=None, data=None, map_est=None,
                 model=None, start=None):
        self.coef = coef
        self.cov = cov
        self.data = data
        self.map_est = map_est
        self.model = model
        self.start = start

    def __str__(self):
        with pd.option_context('display.float_format', '{:.4f}'.format):
            # remove "dtype: object" line from the Series repr
            meanstr = repr(self.coef).rsplit('\n', 1)[0]

        return f"""Quadratic Approximate Posterior Distribution

Formula:
{self.model.str_repr()}

Posterior Means:
{meanstr}
"""

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__str__()}>"

    def sample(self, N=10_000):
        """Sample the posterior approximation."""
        posterior = stats.multivariate_normal(mean=self.coef, cov=self.cov)
        return pd.DataFrame(posterior.rvs(N), columns=self.coef.index)


def quap(vars=None, var_names=None, model=None, data=None, start=None):
    """Compute the quadratic approximation for the MAP estimate.

    Parameters
    ----------
    vars : list, optional, default=model.unobserved_RVs
        List of variables to optimize and set to optimum
    var_names : list, optional
        List of `str` of variables names specified by `model`
    model : pymc.Model (optional if in `with` context)
    start : `dict` of parameter values, optional, default=`model.initial_point`

    Returns
    -------
    result : dict
        Dictionary of `scipy.stats.rv_frozen` distributions corresponding to
        the MAP estimates of `vars`.
    """
    if model is None:
        model = pm.modelcontext(None)

    if vars is None:
        if var_names is None:
            # filter out internally used variables
            mvars, var_names = zip(*[(x, x.name) for x in model.unobserved_RVs
                                     if not x.name.endswith('__')])
        else:
            mvars = [model[x] for x in var_names]
    else:
        if var_names is not None:
            warnings.warn("`var_names` and `vars` set, ignoring `var_names`.")
        mvars = vars
        var_names = [x.name for x in mvars]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        map_est = pm.find_MAP(start=start, vars=mvars, model=model)

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
            # warnings.warn(f"Hessian for '{v.name}' may be incorrect!")
            continue

    # Build output structure
    quap = Quap()

    # Filter variables for output
    basic_vars = set(model.basic_RVs) - set(model.observed_RVs)
    basics = {x.name: x for x in basic_vars}
    deter_vars = set(model.unobserved_RVs) - set(model.basic_RVs)
    dnames = [x.name for x in deter_vars]

    # If requested variables are not basic, just return all of them
    if not set(mvars).intersection(set(basic_vars)):
        var_names = basics.keys()

    cnames = []
    hnames = []
    cvals = []
    for v in basics:
        if v in var_names:
            x = map_est[v]
            if x.size == 1:
                cnames.append(v)
                cvals.append(float(x))
                hnames.append(v)
            elif x.size > 1:
                fmt = '02d' if x.size > 10 else 'd'
                cnames.extend([f"{v}__{k:{fmt}}" for k in range(len(x))])
                cvals.extend(x)
                hnames.append(v)

    # Coefficients are just the basic RVs, without the observed RVs
    quap.coef = pd.Series({x: v for x, v in zip(cnames, cvals)}).sort_index()
    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    H = pm.find_hessian(map_est, vars=[model[x] for x in hnames], model=model)
    quap.cov = (pd.DataFrame(linalg.inv(H), index=cnames, columns=cnames)
                  .sort_index(axis=0) .sort_index(axis=1))
    quap.std = pd.Series(np.sqrt(np.diag(quap.cov)), index=cnames).sort_index()
    quap.map_est = {k: map_est[k] for k in dnames}
    quap.model = model
    quap.start = model.initial_point if start is None else start
    return quap


def sample_quap(quap, N=1000):
    """Return a DataFrame with samples of the posterior in `quap`."""
    return quap.sample(N)


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


def design_matrix(x, poly_order=0, include_const=True):
    """Return array of polynomial weight vectors.

    Parameters
    ----------
    x : (M,) array_like
        Vector of polynomial inputs.
    poly_order : int
        Highest-ordered term exponent.
    include_const : bool
        If True, include a first column of all ones.

    Returns
    -------
    result : (M, poly_order+1) ndarray
        A Vandermonde matrix of increasing powers of `x`.
    """
    x = np.asarray(x)
    try:
        out = np.vander(x, poly_order+1, increasing=True)
    except ValueError:
        raise ValueError(f"poly_order value '{poly_order}' is invalid")
    if not include_const:
        out = out[:, 1:]
    return out


def pad_knots(knots, k=3):
    """Repeat first and last knots `k` times."""
    knots = np.asarray(knots)
    return np.concatenate([np.repeat(knots[0], k),
                           knots,
                           np.repeat(knots[-1], k)])


def bspline_basis(t, x=None, k=3, padded_knots=False):
    """Create the B-spline basis matrix of coefficients.

    Parameters
    ----------
    t : array_like, shape (n+k+1,)
        internal knots
    x : array_like, optional
        points at which to evaluate the B-spline bases. If `x` is not given,
        a `scipy.interpolate.BSpline` object will be returned.
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
