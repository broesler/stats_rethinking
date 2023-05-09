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


# TODO
# * quantile and HPDI (via az.hdi) return transposes of each other for
#   multi-dimensional inputs. Pick one or the other.
# * HPDI does not currently accept multiple q values, but only because the
#   printing function is broken.

def quantile(data, q=0.89, width=6, precision=4,
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
#   * remove dependence on input type. pd.DataFrame.from_dict? or kwarg?
#       R version uses a LOT of "setMethod" calls to allow function to work
#       with many different datatypes.
#       See: <https://github.com/rmcelreath/rethinking/blob/master/R/precis.r>
#
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
        obj = obj.select_dtypes(include=np.number)
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

    def rename(self, mapper):
        """Rename a parameter.

        .. note:: Does NOT work on vector parameters, e.g., 'b__0'.
        """
        self.coef = self.coef.rename(mapper)
        self.cov = self.cov.rename(index=mapper, columns=mapper)
        self.std = self.std.rename(mapper)


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
        map_est = pm.find_MAP(start=start,
                              vars=mvars,
                              progressbar=False,
                              model=model)

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
    basic_vars = model.free_RVs
    dnames = [x.name for x in model.deterministics]

    # If requested variables are not basic, just return all of them
    out_vars = set(mvars).intersection(set(basic_vars))
    if not out_vars:
        out_vars = basic_vars

    cnames = []
    hnames = []
    cvals = []
    for ov in out_vars:
        v = ov.name
        x = map_est[v]
        if x.size == 1:
            cnames.append(v)
            cvals.append(float(x))
            hnames.append(v)
        elif x.size > 1:
            # TODO case of 2D, etc. variables
            # Flatten vectors into singletons 'b__0', 'b__1', ..., 'b__n'
            fmt = '02d' if x.size > 10 else 'd'
            cnames.extend([f"{v}__{k:{fmt}}" for k in range(len(x))])
            cvals.extend(x)
            hnames.append(v)  # need the root name for Hessian

    # TODO store coefficients as an xarray Dataset to accomodate
    # multi-dimensional parameters? Having flat Series/DataFrame for means and
    # covariance matrix makes using stats.multivariate_normal simple, but then
    # user-code needs to do the "unflattening" to combine [alpha, b0, b1, ...]
    # for mathematical computations.

    # Coefficients are just the basic RVs, without the observed RVs
    quap.coef = pd.Series({x: v for x, v in zip(cnames, cvals)}).sort_index()
    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    H = pm.find_hessian(map_est, vars=[model[x] for x in hnames], model=model)
    quap.cov = (pd.DataFrame(linalg.inv(H), index=cnames, columns=cnames)
                  .sort_index(axis=0).sort_index(axis=1))
    quap.std = pd.Series(np.sqrt(np.diag(quap.cov)), index=cnames).sort_index()
    quap.map_est = {k: map_est[k] for k in dnames}
    quap.model = model
    quap.start = model.initial_point if start is None else start
    quap.data = data  # FIXME need to pass data for each call of quap!!
    return quap


# TODO
# * usage with "with model: pm.set_data(...)" vs passing in
#   a DataFrame + an `eval_at` string
# * how to use pymc on_unused_input='warn' so we can just pass all
#   variables to the model.[var].eval() call and not have to specify which
#   output gets which inputs.
# * (un)flatten list of vector or matrix parameters
#   See: the_model.eval_rv_shapes()
# * rename 'eval_lm'?
#
def lmeval(fit, out='mu', params=None, eval_at=None, dist=None):
    """Sample the indermediate linear models from `the_model`."""
    pm.set_data(eval_at, model=fit.model)

    if dist is None:
        dist = fit.sample()  # take the posterior

    # Could use this to determine the Deterministic RVs if none specified,
    # and loop over each output variable.
    # The issue with this method is that the *inputs* to each would need to
    # be determined by traversing the pytensor graph?
    # out_vars = model.deterministics
    out_vars = [x for x in fit.model.unobserved_RVs if x.name == out]
    if out_vars:
        out_var = out_vars[0]
    else:
        raise ValueError(f"Variable '{out}' does not exist in the model!")

    param_vars = [x for x in fit.model.unobserved_RVs if x.name in params]

    # Manual loop since params are 0-D variables in the model.
    Ne = np.max([x.size for x in eval_at.values()])
    out_s = np.zeros((Ne, len(dist)))
    for i in range(len(dist)):
        param_vals = {v: dist.loc[i, v.name] for v in param_vars}
        out_s[:, i] = out_var.eval(param_vals)

    return out_s


# TODO
# * add "ci" = {'hpdi', 'pi', None} option
# * use lmeval to compute mu_samp
def lmplot(quap, data, x, y, eval_at=None, unstd=False,
           q=0.89, ax=None):
    """Plot the linear model defined by `quap`.

    Parameters
    ----------
    quap : sts.Quap
        The quadratic approximation model estimate.
    data : DataFrame
        The data used to fit the model.
    x, y : str
        The column names of the data points to plot.
    eval_at : array_like
        The values at which to evaluate the linear model.
    unstd : bool
        If True, the model was fit to standardized values, so un-standardize
        them to plot in coordinates with real units.
    q : float in [0, 1]
        Quartile over which to shade the mean.
    ax : plt.Axes
        Axes object in which to draw the plot.

    Returns
    -------
    ax : plt.Axes
        The axes in which the plot was drawn.
    """
    if eval_at is None:
        eval_at = data[x].sort_values().values
    if ax is None:
        ax = plt.gca()

    post = quap.sample()
    mu_samp = (post['alpha'].values
               + post['beta'].values * eval_at[:, np.newaxis])
    mu_mean = mu_samp.mean(axis=1)
    mu_pi = sts.percentiles(mu_samp, q=q, axis=1)  # 0.89 default

    if unstd:
        eval_at = sts.unstandardize(eval_at, data[x])
        mu_mean = sts.unstandardize(mu_mean, data[y])
        mu_pi = sts.unstandardize(mu_pi, data[y])

    ax.scatter(x, y, data=data, alpha=0.4)
    ax.plot(eval_at, mu_mean, 'C0', label='MAP Prediction')
    ax.fill_between(eval_at, mu_pi[0], mu_pi[1],
                    facecolor='C0', alpha=0.3, interpolate=True,
                    label=rf"{100*q:g}% Percentile Interval of $\mu$")
    ax.set(xlabel=x, ylabel=y)
    return ax


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


def unstandardize(x, data=None, axis=0):
    """Return the data to the original scale."""
    if data is None:
        data = x
    return data.mean(axis=axis) + x * data.std(axis=axis)


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


# TODO make own class?
def coef_table(models, mnames=None, params=None, std=True):
    """Create a summary table of coefficients in each model.

    Parameters
    ----------
    models : list of `Quap`
        The models over which to summarize.
    mnames : list of str, optional
        Names of the models.
    params : list of str, optional
        Names of specific parameters to return.
    std : bool, optional
        If True, also return a table of standard deviations.

    Returns
    -------
    ct, cs : pd.DataFrame
        DataFrames of the coefficients and their standard deviations.
    """
    coefs = [m.coef for m in models]
    stds = [m.std for m in models]

    def transform_ct(ct, mnames, params=None, value_name='coef'):
        """Make coefficient table tidy for plotting"""
        ct.columns = mnames
        ct.index.name = 'param'
        ct.columns.name = 'model'
        if params is not None:
            ct = ct.loc[params]
        ct = (ct.T  # organize by parameter, then model
                .melt(ignore_index=False, value_name=value_name)
                .set_index('param', append=True)
                .sort_index()
              )
        return ct

    ct = transform_ct(pd.concat(coefs, axis=1), mnames, params)
    if not std:
        return ct

    cs = transform_ct(pd.concat(stds, axis=1), mnames, params,
                      value_name='std')
    return pd.concat([ct, cs], axis=1)


def plot_coef_table(ct, q=0.89, ax=None):
    """Plot the table of coefficients from `sts.coef_table`.

    Parameters
    ----------
    ct : :obj:`CoefTable`
        Coefficient table output from `coef_table`.
    q : float in [0, 1], optional
        The probability interval to plot.
    ax : Axes, optional
        The Axes on which to plot.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        ax = plt.gca()

    # Leverage Seaborn for basic setup
    sns.pointplot(data=ct.reset_index(), x='coef', y='param', hue='model',
                  join=False, dodge=0.2)

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    colors = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            if not np.ma.is_masked(x) and not np.ma.is_masked(y):
                x_coords.append(x)
                y_coords.append(y)
                colors.append(point_pair.get_facecolor())

    # Manually add the errorbars since we have std values already
    z = stats.norm.ppf(1 - (1 - q)/2)
    errs = 2 * ct['std'] * z
    errs = errs.dropna()
    ax.errorbar(x_coords, y_coords, fmt=' ', xerr=errs, ecolor=colors)
    ax.axvline(0, ls='--', c='k', lw=1, alpha=0.5)
    return ax

# =============================================================================
# =============================================================================
