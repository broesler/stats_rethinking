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
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import sys
import uuid
import warnings
import xarray as xr

from abc import ABC
from arviz.data.base import generate_dims_coords
from contextlib import contextmanager
from copy import deepcopy

from pytensor.graph.basic import ancestors
# from pytensor.tensor.random.var import (
#     RandomGeneratorSharedVariable,
#     RandomStateSharedVariable,
# )
from pytensor.tensor.sharedvar import TensorSharedVariable  # , SharedVariable
from pytensor.tensor.var import TensorConstant, TensorVariable

from scipy import stats, linalg
from scipy.interpolate import BSpline
from scipy.special import logsumexp as _logsumexp
from sparkline import sparkify

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

# TODO
# * define __all__ for the package to declutter internal variables
# * quantile and HPDI (via az.hdi) return transposes of each other for
#   multi-dimensional inputs. Pick one or the other.
# * HPDI does not currently accept multiple q values, but only because the
#   printing function is broken.
# * `pairs` method to plot pair-wise covariance given Quap (sample, then
#   seaborn.pairplot())


def quantile(data, qs=0.89, width=6, precision=4,
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
    quantile : scalar or ndarray
        The requested quantiles. See documentation for `numpy.quantile`.
        Shape = (Q, M), where Q = q.shape, and M is the shape of the data
        without the given axis. If no axis is given, data is unraveled and
        output shape is Q.

    Examples
    --------
    >>> B = np.random.random((5, 10, 7))
    >>> np.quantile(B, q=[1 - 0.89, 0.89]).shape
    === (2,)
    >>> np.quantile(B, q=[1 - 0.89, 0.89], axis=1).shape
    === (2, 5, 7)

    See Also
    --------
    `numpy.quantile`
    """
    qs = np.asarray(qs)
    quantiles = q_func(data, qs, **kwargs)
    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-1}g}%" for p in np.atleast_1d(qs)])
        value_str = ' '.join([f"{q:{fstr}}" for q in np.atleast_1d(quantiles)])
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

    Returns
    -------
    percentiles : ndarray
        The requested high and low quantiles.
        Shape = (2, Q, M), where Q = q.shape, and M is the shape of the given
        axis. If no axis is given, data is unraveled and output shape is Q.
        The low boundary is index 0, and the high boundary is index 1.

    See Also
    --------
    quantile
    """
    a = (1 - (q)) / 2
    quantiles = quantile(data, (a, 1-a), **kwargs)
    return quantiles


# TODO remove width and precision arguments and just take fstr='8.2f', e.g.
# * update docs to show that it matches sts.quantile shapes.
# * add axis=1 argument ->
# * allow multiple qs, but print them "nested" like on an x-axis.
def hpdi(data, q=0.89, verbose=False, width=6, precision=4,
         axis=None, **kwargs):
    """Compute highest probability density interval.

    .. note::
        This function depends on `arviz.hdi`.

    Parameters
    ----------
    data : (M, N) array_like
        Matrix of M vectors in N dimensions
    q : array_like
    verbose : bool
    width : int
    precision : int
    axis : int
        If None, unravel the data.
    kwargs : dict_like

    Returns
    -------
    quantiles : (M, N) ndarray
        Matrix of M vectors in N dimensions

        data is (M, N, P)
        az.hdi(data, hdi_prob=0.89).shape == (P, 2)
        np.array([az.hdi(data, hdi_prob=q) for q in qs]).shape == (Q, P, 2)
        ==> az.hdi "ravels" the array along the first 2 dimensions.

    Examples
    --------
    >>> A = np.random.random((5, 10))
    >>> az.hdi(A, hdi_prob=0.89).shape
    === (10, 2)  # FutureWarning (draw, shape) -> (chain, draw)
    >>> B = np.random.random((5, 10, 7))
    >>> az.hdi(B, hdi_prob=0.89).shape
    === (7, 2)
    >>> C = np.random.random((5, 10, 7, 4))
    >>> az.hdi(C, hdi_prob=0.89).shape
    === (7, 4, 2)


    """
    qs = np.asarray(q)
    data = np.asarray(data)

    if verbose and qs.size > 1:
        verbose = False
        warnings.warn("verbose flag only valid for singleton q.")

    if axis is None:
        A = data.reshape(-1).squeeze()  # (M, N) -> (M * N,)
    else:
        # az.hdi "ravels" the array along the first 2 dimensions, because it
        # assumes that the data is from az.convert_to_data() which returns an
        # xarray with (chain, draw, ...) dimensions.
        # ==> move the desired axis to the front, then add 1 dimension.
        # (M, *N*, P) -> (*N*, M, P) -> (1, *N*, M, P)
        A = np.moveaxis(data, axis, 0)[np.newaxis, :]

    # (1, N, M, P) -> (M, P, 2) -> (Q, M, P, 2)
    if qs.ndim == 0:
        Q = az.hdi(A, hdi_prob=qs, **kwargs)
    else:
        Q = np.stack([az.hdi(A, hdi_prob=q, **kwargs) for q in qs])

    Q = np.moveaxis(Q, -1, 0)  # (Q, M, P, 2) -> (2, Q, M, P)

    if verbose:
        fstr = f"{width}.{precision}f"
        name_str = ' '.join([f"{100*p:{width-2}g}%" for p in np.r_[qs, qs]])
        value_str = ' '.join([f"{q:{fstr}}" for q in np.atleast_1d(Q)])
        print(f"|{name_str}|\n{value_str}")

    return Q


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
    return pd.DataFrame(itertools.product(*kwargs.values()),
                        columns=kwargs.keys())


# TODO
#   * expand documentation with examples
#   * remove dependence on input type. pd.DataFrame.from_dict? or kwarg?
#       R version uses a LOT of "setMethod" calls to allow function to work
#       with many different datatypes.
#       See: <https://github.com/rmcelreath/rethinking/blob/master/R/precis.r>
#       pythonic way would be to make objects that contain a precis method.
#
def precis(obj, p=0.89, digits=4, verbose=True, hist=True):
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
    if not isinstance(obj, (PostModel, xr.Dataset, pd.DataFrame, np.ndarray)):
        raise TypeError(f"`obj` of type '{type(quap)}' is unsupported!")

    a = (1-p)/2
    pp = 100*np.array([a, 1-a])  # percentages for printing

    if isinstance(obj, PostModel):
        title = None
        # Compute density intervals
        coef = dataset_to_series(obj.coef)
        z = stats.norm.ppf(1 - a)
        lo = coef - z * obj.std
        hi = coef + z * obj.std
        df = pd.concat([coef, obj.std, lo, hi], axis=1)
        df.columns = ['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
        # if hist:
        #     df['histogram'] = sparklines_from_norm(df['mean'], df['std'])

    # Dataset of data points (i.e. posterior distribution)
    if isinstance(obj, xr.Dataset):
        if 'draw' not in obj.dims:
            raise TypeError("Expected dimensions ['draw'] in `obj`")
        if 'chain' in obj.dims:
            obj = obj.mean('chain')
        title = (f"'DataFrame': {obj.sizes['draw']:d} obs."
                 f" of {len(obj.data_vars)} variables:")
        mean = dataset_to_series(obj.mean('draw'))
        std = dataset_to_series(obj.std('draw'))
        z = stats.norm.ppf(1 - a)
        lo = mean - z * std
        hi = mean + z * std
        df = pd.concat([mean, std, lo, hi], axis=1)
        df.columns = ['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
        if hist:
            df['histogram'] = sparklines_from_dataframe(dataset_to_frame(obj))

    # DataFrame of data points
    if isinstance(obj, pd.DataFrame):
        obj = obj.select_dtypes(include=np.number)
        title = f"'DataFrame': {obj.shape[0]:d} obs. of {obj.shape[1]} variables:"
        df = pd.DataFrame()
        df['mean'] = obj.mean()
        df['std'] = obj.std()
        for i in range(2):
            df[f"{pp[i]:g}%"] = obj.apply(lambda x: np.nanpercentile(x, pp[i]))
        if hist:
            df['histogram'] = sparklines_from_dataframe(obj)

    # Numpy array of data points
    if isinstance(obj, np.ndarray):
        title = f"'ndarray': {obj.shape[0]:d} obs. of {obj.shape[1]} variables:"
        # Columns are data, ignore index
        vals = np.vstack([np.nanmean(obj, axis=0),
                          np.nanstd(obj, axis=0),
                          np.nanpercentile(obj, pp[0], axis=0),
                          np.nanpercentile(obj, pp[1], axis=0)]).T
        df = pd.DataFrame(vals,
                          columns=['mean', 'std', f"{pp[0]:g}%", f"{pp[1]:g}%"]
                          )
        if hist:
            df['histogram'] = sparklines_from_array(obj)

    if verbose:
        if title is not None:
            print(title)
        # Print the dataframe with requested precision
        with pd.option_context('display.float_format',
                               f"{{:.{digits}f}}".format):
            print(df)

    return df


def sparklines_from_norm(means, stds, width=12):
    """Generate list of sparklines from means and stds."""
    # Create matrix of samples
    assert len(means) == len(stds)
    Nm = len(means)
    Ns = 1000
    samp = stats.norm(np.c_[means], np.c_[stds]).rvs(size=(Nm, Ns))
    sparklines = []
    for s in samp:
        sparklines.append(sparkify(np.histogram(s, bins=width)[0]))
    return sparklines


# TODO sparklines_from_dataset fails for N-D variables.
def sparklines_from_dataset(ds, width=12):
    """Generate list of sparklines from a Dataset."""
    if 'draw' not in ds.dims:
        raise TypeError("Expected dimension 'draw' in `ds`")
    sparklines = []
    for v in ds.data_vars:
        data = ds[v].dropna(dim='draw')
        sparklines.append(sparkify(np.histogram(data, bins=width)[0]))
    return sparklines


def sparklines_from_dataframe(df, width=12):
    """Generate list of sparklines from a DataFrame."""
    sparklines = []
    for col in df:
        data = df[col].dropna()
        sparklines.append(sparkify(np.histogram(data, bins=width)[0]))
    return sparklines


def sparklines_from_array(arr, width=12):
    """Generate list of sparklines from an array of data."""
    sparklines = []
    for col in arr.T:
        data = col[np.isfinite(col)]
        sparklines.append(sparkify(np.histogram(data, bins=width)[0]))
    return sparklines


# -----------------------------------------------------------------------------
#         Quap functions
# -----------------------------------------------------------------------------
# TODO
# * make all attributes read-only? quap() call populates struct.
# * can require kwargs on __init__, then use those values to compute self._std,
#   etc. so that the property just returns that value without doing
#   a computation each time it is called.
# * move `map_est` to Quap only.

class PostModel(ABC):
    """
    Attributes
    ----------
    coef : dict
        Dictionary of maximum *a posteriori* (MAP) coefficient values.
    cov : (M, M) DataFrame
        Covariance matrix of the parameters.
    std : (M,) Series
        Standard deviation. The square root of the diagonal of `cov`.
    data : (M, N) array_like
        Matrix of the data used to compute the likelihood.
    map_est : dict
        Maximum *a posteriori* estimates of any Deterministic or Potential
        variables.
    loglik : float
        The minus log-likelihood of the data, given the model parameters.
    model : :class:`pymc.Model`
        The pymc model object used to define the posterior.
    start : dict
        Initial parameter values for the MAP optimization. Defaults to
        `model.initial_point`.
    """
    _descrip = ""

    def __init__(self, *, coef=None, cov=None, data=None, map_est=None,
                 loglik=None, model=None, start=None):
        self.coef = coef
        self.cov = cov
        self.data = data
        self.map_est = map_est
        self.loglik = loglik
        self.model = model
        self.start = start

    @property
    def std(self):
        return pd.Series(np.sqrt(np.diag(self.cov)), index=self.cov.index)

    @property
    def corr(self):
        D = pd.DataFrame(
            np.diag(1 / self.std),
            index=self.cov.index,
            columns=self.cov.columns,
        )
        return D @ self.cov @ D

    def get_samples(self):
        """Get samples from the posterior distribution.

        .. note::
            For quadratic approximations, these samples will be drawn with this
            function call. For MCMC models, the samples will already be stored
            in the object, so this function just returns them.
        """
        pass

    def sample_prior(self, N=10_000):
        """Sample the prior distribution.

        Analagous to `rethinking::extract.prior`.
        """
        idata = pm.sample_prior_predictive(samples=N, model=self.model)
        return idata.prior.mean('chain')

    def deviance(self):
        """Return the deviance of the model."""
        return -2 * self.loglik

    def AIC(self):
        """Return the Akaike information criteria of the model."""
        return self.deviance() + 2*sum(self.coef.sizes.values())

    # TODO
    # * rename the model variable itself
    #   now:
    #   >>> model.named_vars
    #   === {'x': x, 'y': y, 'z': z}
    #   >>> model.named_vars['x'].name = 'new_name'
    #   >>> model.named_vars
    #   === {'x': new_name, 'y': y, 'z': z}
    #   Want the key 'x' to be changed to 'new_name' as well.
    # * rename any vector parameters 'b__0', 'b__1', etc.
    def rename(self, mapper):
        """Rename a parameter.

        .. note:: Does NOT work on vector parameters, e.g., 'b__0'.
        """
        self.coef = self.coef.rename(mapper)
        self.cov = self.cov.rename(index=mapper, columns=mapper)
        for k, v in mapper.items():
            self.model.named_vars[k].name = v
        return self

    # TODO allow subclasses to provide "description" line in `out`?
    def __str__(self):
        with pd.option_context('display.float_format', '{:.4f}'.format):
            try:
                # remove "dtype: object" line from the Series repr
                meanstr = repr(self.coef.to_pandas()).rsplit('\n', 1)[0]
            except ValueError:
                meanstr = repr(self.coef)

        # FIXME indentation. inspect.cleandoc() fails because the
        # model.str_repr() is not always aligned left.
        out = f"""{self._descrip}

Formula:
{self.model.str_repr()}

Posterior Means:
{meanstr}

Log-likelihood: {self.loglik:.2f}
"""
        return out

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__str__()}>"


class Quap(PostModel):
    _descrip = "Quadratic-approximate posterior"
    __doc__ = _descrip + "\n" + PostModel.__doc__

    def sample(self, N=10_000):
        """Sample the posterior approximation.

        Analagous to `rethinking::extract.samples`.
        """
        mean = flatten_dataset(self.coef).values
        posterior = stats.multivariate_normal(mean=mean, cov=self.cov)
        df = pd.DataFrame(posterior.rvs(N), columns=self.cov.index)
        return frame_to_dataset(df, model=self.model).mean('chain')

    def get_samples(self, N=1000):
        return self.sample(N)


class Ulam(PostModel):
    _descrip = "Hamiltonian Monte Carlo approximation."
    __doc__ = _descrip + "\n" + PostModel.__doc__

    def __init__(self, samples=None, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples

    def get_samples(self, N=1000):
        # TODO currently ignoring `N` argument.
        return self.samples

    # NOTE unlike rethinking::traceplot, pymc discards the warmup samples by
    # default, and only returns the valid samples. We could write a function
    # that mimics rethinking::traceplot, using:
    # >>> idata = pm.sample(..., discard_tuned_samples=False)
    # >>> all_post = xr.concat([idata.warmup_posterior, idata.posterior],
    #                          dim='concat_dim')
    # >>> az.plot_trace(all_post) 
    # or something to that effect.
    #
    def plot_trace(self, title=None):
        """Plot the MCMC sample chains for each parameter.

        Parameters
        ----------
        title : str, optional
            The title of the figure.

        Returns
        -------
        fig : plt.Figure
            The figure handle containing the trace plots.
        axes : ndarray of plt.Axes
            An array corresponding to the axes of each trace plot.
        """
        p = az.plot_trace(self.samples)
        fig = p[0, 0].figure
        fig.suptitle(title)
        return fig, p

    def pairplot(self, title=None, **kwargs):
        """Plot the pairwise correlations between the model parameters.

        Parameters
        ----------
        title : str, optional
            The title of the figure.
        kwargs : dict, optional
            Additional arguments to be passed to `seaborn.pairplot()`.

        Returns
        -------
        grid : seaborn.PairGrid
            Returns the underlying instance for further tweaking.
        """
        opts = dict(
            corner=True,
            diag_kind='kde',
            plot_kws=dict(s=10, alpha=0.2),
            height=1.5,
        )
        if kwargs is not None:
            opts.update(kwargs)
        g = sns.pairplot(dataset_to_frame(self.samples), **opts)
        g.figure.suptitle(title)
        return g


def quap(vars=None, var_names=None, model=None, data=None, start=None):
    """Compute the quadratic approximation for the MAP estimate.

    Parameters
    ----------
    vars : list, optional, default=model.unobserved_RVs
        List of variables to optimize and set to optimum.
    var_names : list, optional
        List of `str` of variables names specified by `model`. If `vars` is
        given, `var_names` will be ignored.
    model : pymc.Model (optional if in `with` context)
    data : pd.DataFrame, optional
        The data to which this model was fit.
    start : `dict` of parameter values, optional, default=`model.initial_point`

    Returns
    -------
    result : Quap
        Object containing information about the posterior parameter values.
    """
    model = pm.modelcontext(model)

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
        # Ignore these warnings:
        # "UserWarning: Intermediate variables (such as Deterministic or
        # Potential) were passed. find_MAP will optimize the underlying
        # free_RVs instead."
        warnings.simplefilter('ignore', category=UserWarning)
        map_est, opt = pm.find_MAP(
                start=start,
                vars=mvars,
                return_raw=True,
                progressbar=False,
                model=model
            )

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

    # TODO clean up and get rid of these temp variables
    # Filter variables for output
    free_vars = model.free_RVs
    dnames = [x.name for x in model.deterministics]

    # If requested variables are not free, just return all of them
    out_vars = set(mvars).intersection(set(free_vars)) or free_vars
    out_vars = sorted(out_vars, key=lambda x: x.name)

    # Convert coefficients to an xarray.Dataset to retain variable shape info
    coef = dict_to_dataset({v.name: map_est.get(v.name) for v in out_vars})
    cnames = cnames_from_dataset(coef)

    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    H = pm.find_hessian(map_est, vars=tuple(out_vars), model=model)

    # Coefficients are just the basic RVs, without the observed RVs
    return Quap(
        coef=coef,
        cov=pd.DataFrame(linalg.inv(H), index=cnames, columns=cnames),
        data=deepcopy(data),
        map_est={k: map_est[k] for k in dnames},
        loglik=opt.fun,  # equivalent of sum(loglik(model, pointwise=False))
        model=deepcopy(model),
        start=model.initial_point() if start is None else start,
    )


# TODO
# * remove mean('chain') calls. These should be converted to:
#       ds = ds.stack(sample=('chain', 'draw')).transpose('sample', ...)
# * get desired number of samples from chains?
def ulam(vars=None, var_names=None, model=None, data=None, start=None, **kwargs):
    """Compute the quadratic approximation for the MAP estimate.

    Parameters
    ----------
    vars : list of TensorVariables, optional, default=model.free_RVs
        List of variables to optimize and set to optimum.
    var_names : list of str, optional
        List of `str` of variables names specified by `model`.
    model : pymc.Model (optional if in `with` context)
    data : pd.DataFrame, optional
        The data to which this model was fit.
    start : dict[str] -> ndarray, optional, default=`model.initial_point`
        Dictionary of initial parameter values. Keys should be names of random
        variables in the model.
    **kwargs : dict
        Additional argumements to be passed to `pymc.sample()`.

    Returns
    -------
    result : Ulam
        Object containing information about the posterior parameter values.
    """
    sample_dims = ('chain', 'draw')
    model = pm.modelcontext(model)
    idata = pm.sample(
        model=model,
        initvals=start,
        idata_kwargs=dict(log_likelihood=True),
        **kwargs
    )

    # Get requested variables
    if vars is None and var_names is None:
        # filter out internally used variables
        var_names = [x.name for x in model.free_RVs
                     if not x.name.endswith('__')]
    else:
        if var_names is not None:
            warnings.warn("`var_names` and `vars` set, ignoring `var_names`.")
        var_names = [x.name for x in vars]

    # Get the posterior samples
    post = idata.posterior[list(var_names)]

    # Coefficient values are just the mean of the samples
    coef = post.mean(sample_dims)

    # Compute the covariance matrix from the samples
    cov = dataset_to_frame(
        post
        .stack(sample=sample_dims)
        .transpose('sample', ...)
    ).cov()

    # Get the minus log likelihood of the data
    loglik = float(
        -idata
        .log_likelihood
        .mean(sample_dims)
        .sum()
        .to_dataarray()  # convert to be able to extract singleton value
    )

    # Coefficients are just the basic RVs, without the observed RVs
    return Ulam(
        coef=coef,
        cov=cov,
        data=deepcopy(data),
        loglik=loglik,
        model=deepcopy(model),
        start=model.initial_point() if start is None else start,
        samples=post,
    )


# TODO
# * Idea: could create a similar API to pymc.sample_prior_predictive, etc. that
#   is a wrapper on this function to sample the Deterministics or Observed
#   variables as appropriate, and returns an inference data object instead of
#   a basic numpy array.
#
# * `rethinking::sim` equivalent:
#   - For each Observed variable,
#       - get its inputvars
#       - remove Determinstic vars from inputvars
#       - compile_fn and apply with the posterior samples
#
# * Loop over each Deterministic and Observed variables if none specified,
#   return a dict. Then, lmplot can plot both the mean and the observed
#   confidence intervals in one shot.
# * (un)flatten list of vector or matrix parameters
#   See: the_model.eval_rv_shapes()
# * keep_data=True? Need to get model data for each `eval_at.keys()`
# * refactor `out` to `mean_var` or `lm_var`?
#
def lmeval(fit, out, params=None, eval_at=None, dist=None, N=1000):
    """Sample the indermediate linear models from the given parameter fit.

    .. note:: This function is similar to `rethinking::link`, but with a more
        readable name.

    Parameters
    ----------
    fit : :class:`stats_rethinking.Quap` or similar
        An object containing a pymc model and a posterior distribution.
    out : TensorVariable
        The output variable corresponding to the linear model.
    params : list of TensorVariables
        The parameters of the linear model. If not all parameters are
        specified, the values will be determined by the state of the random
        number generator in the PyTensor graph.
    eval_at : dict like {str: array_like}
        A dictionary of the independent variables over which to evaluate the
        model. Keys must be strings of variable names, and values must be
        array-like, with dimension equivalent to the corresponding variable
        definition in the model.
    dist : dict or DataFrame, default None.
        A dict or DataFrame containing samples of the distribution of the
        `params` as values/columns. If `dist` is None, the posterior
        distribution will be used.
    N : int
        If `dist` is None, number of samples to take from `dist`.

    Returns
    -------
    samples : (M, N) ndarray
        An array of values of the linear model evaluated at each of M `eval_at`
        points and `N` parameter samples.
    """
    if out.name not in fit.model:
        raise ValueError(f"Variable '{out}' does not exist in the model!")

    if params is None:
        params = inputvars(out)

    if dist is None:
        dist = fit.sample(N)  # take the posterior

    if eval_at is not None:
        pm.set_data(eval_at, model=fit.model)

    # Compile the graph function to compute. Better than `eval`, which
    # does *not* generate a new random state for each call.
    out_func = fit.model.compile_fn(
        inputs=params,
        outs=out,
        on_unused_input='ignore',
    )

    # Manual loop since params are 0-D variables in the model.
    out_samp = np.fromiter(
        (
            out_func({v.name: dist[v.name].isel(draw=i) for v in params})
            for i in range(dist.sizes['draw'])
        ),
        dtype=np.dtype((float, out.shape.eval())),
        count=dist.sizes['draw'],
    ).T  # (out.shape, draw)

    # Return a DataArray for named dimensions.
    return xr.DataArray(
        out_samp,
        coords={
            f"{out.name}_dim_0": range(out_samp.shape[0]),
            'draw': range(out_samp.shape[1])
        }
    )


# TODO
# * options for `unstd` in {'x', 'y', 'both'}
# * add "ci" = {'hpdi', 'pi', None} option
# * add option for observed variable and plots its PI too.
def lmplot(quap=None, mean_var=None, fit_x=None, fit_y=None,
           x=None, y=None, data=None,
           eval_at=None, unstd=False, q=0.89, ax=None,
           line_kws=None, fill_kws=None, marker_kws=None,
           label='MAP Prediction'):
    """Plot the linear model defined by `quap`.

    Parameters
    ----------
    fit_x : (N,) array-like
        1-D array of values over which to plot the model fit.
    fit_y : (N, S) array-like
        A 2-D array of `S` samples corresponding to the `N` `fit_x` points.
        .. note:: Either `fit_x` and `fit_y`, or `quap` and `mean_var` must be
            provided to plot the fitted model.
    quap : stats_rethinking.Quap
        The quadratic approximation model estimate.
    mean_var : TensorVariable
        The variable corresponding to the linear model of the mean.
    data : DataFrame
        The data used to fit the model.
    x, y : str
        The column names of the data points to plot.
    eval_at : dict of str: array_like
        A dictionary of the independent variables over which to evaluate the
        model. Keys must be strings of variable names, and values must be
        array-like, with dimension equivalent to the corresponding variable
        definition in the model.
    unstd : bool
        If True, the model was fit to standardized values, so un-standardize
        them to plot in coordinates with real units.
    q : float in [0, 1]
        Quartile over which to shade the mean.
    ax : plt.Axes
        Axes object in which to draw the plot.
    line_kws, fill_kws : dict
        Maps of additional arguments for the line or fill plots.

    Returns
    -------
    ax : plt.Axes
        The axes in which the plot was drawn.
    """
    if ((fit_x is None and fit_y is None)
            and (quap is None and mean_var is None)):
        raise ValueError('Must provide either fit_[xy] or (quap, mean_var)!')

    if ax is None:
        ax = plt.gca()

    if line_kws is None:
        line_kws = dict()

    if fill_kws is None:
        fill_kws = dict()

    if marker_kws is None:
        marker_kws = dict()

    if quap is not None and mean_var is not None:
        # TODO? remove ALL of this nonsense and just require mu_samp as input.
        # User code then calls:
        #   mu_s = lmeval()
        #   lmplot(mu_s, ...)
        data_vars = set(named_graph_inputs([mean_var])) - set(inputvars(mean_var))
        data_names = [v.name for v in data_vars]

        if eval_at is None:
            # Use the given data to evaluate the model
            xe = data[x].sort_values()
            if unstd:
                xe = standardize(xe)

            # Determine which name to use
            if len(data_vars) == 1:
                eval_at = {data_names[0]: xe}
            elif x in data_names:
                eval_at = {x: xe}
            elif len(data_vars) > 1:
                raise ValueError("More than 1 data variable in the model!"
                                 + " Please specify `eval_at`")
        else:
            if len(eval_at) == 1:
                xe = list(eval_at.values())[0]
            elif x in eval_at:
                xe = eval_at[x]
            else:
                raise ValueError(f"Variable '{x}' not found in model.")

        # Ensure the passed-in variable names match those in the model.
        # If the user has not done quap.rename(), this step should be a no-op.
        name_map = {v.name: k for k, v in quap.model.named_vars.items()}
        eval_rename = dict()
        for k, v in eval_at.items():
            eval_rename[name_map[k]] = v
        eval_at = eval_rename

        mu_samp = lmeval(quap, out=mean_var, eval_at=eval_at)

    elif fit_x is not None and fit_y is not None:
        xe = fit_x
        mu_samp = fit_y

    # TODO expect xarray with dimension 'draw'
    # Compute mean and error
    mu_mean = mu_samp.mean(axis=1)
    mu_pi = percentiles(mu_samp, q=q, axis=1)  # 0.89 default

    if unstd:
        xe = unstandardize(xe, data[x])
        mu_mean = unstandardize(mu_mean, data[y])
        mu_pi = unstandardize(mu_pi, data[y])

    # TODO update "pop" calls by defining default dicts, then d.update(kwargs)?
    # Make the plot
    if data is not None:
        ax.scatter(
            x, y, data=data,
            alpha=marker_kws.pop('alpha', 0.4),
            **marker_kws
        )
    ax.plot(xe, mu_mean, label=label,
            c=line_kws.pop('color', line_kws.pop('c', 'C0')), **line_kws)
    ax.fill_between(xe, mu_pi[0], mu_pi[1],
                    facecolor=fill_kws.pop('facecolor', 'C0'),
                    alpha=fill_kws.pop('alpha', 0.3),
                    interpolate=True,
                    label=rf"{100*q:g}% Percentile Interval",
                    **fill_kws)
    ax.set(xlabel=x, ylabel=y)
    return ax


# -----------------------------------------------------------------------------
#         Graph Utilities
# -----------------------------------------------------------------------------
def named_graph_inputs(graphs, blockers=None):
    """Return list of inputs to a PyTensor variable.

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `variable`s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
        Input nodes with a name, in the order found by a left-recursive
        depth-first search started at the nodes in `graphs`.

    See Also
    --------
    pytensor.graph.basic.graph_inputs
    <https://github.com/pymc-devs/pytensor/blob/095e3c9b05525583d3c5a98f9bb75eb6f7ca4556/pytensor/graph/basic.pyL882-L902>
    """
    graphs_names = [v.name for v in graphs]
    yield from (r
                for r in ancestors(graphs, blockers)
                if r.name is not None and r.name not in graphs_names
                )


def inputvars(a):
    """Get the inputs to PyTensor variables.

    Parameters
    ----------
        a: PyTensor variable

    Returns
    -------
        r: list of tensor variables that are inputs

    See Also
    --------
    pymc.inputvars
    <https://www.pymc.io/projects/docs/en/stable/_modules/pymc/pytensorf.html#inputvars>
    """
    return [
        v
        for v in named_graph_inputs(_makeiter(a))
        if isinstance(v, TensorVariable) and not isinstance(v, TensorConstant)
    ]


def _makeiter(a):
    """Return an iterable of the input."""
    return a if isinstance(a, (tuple, list)) else [a]


# -----------------------------------------------------------------------------
#         Model-building
# -----------------------------------------------------------------------------
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


def standardize(x):
    """Standardize the input vector `x` by the mean and std of `data`.

    .. note::
        The following lines are equivalent:
                           (x - x.mean()) / x.std() == stats.zscore(x, ddof=1)
        (N / (N-1))**0.5 * (x - x.mean()) / x.std() == stats.zscore(x, ddof=0)
        where N = x.size
    """
    center = x.mean()
    scale = x.std()
    z = (x - center) / scale
    if hasattr(z, 'attrs'):
        z.attrs = {'center': center, 'scale': scale}
    return z


def unstandardize(x, data=None):
    """Return the data to the original scale.

    Parameters
    ----------
    x : array_like
        The scaled data.
    data : array_like
        The un-scaled data with which to compute the center and scale.

    Returns
    -------
    result : un-scaled array of the same size as `x`.
    """
    if data is None:
        try:
            center = x.attrs['center']
            scale = x.attrs['scale']
        except KeyError:
            raise ValueError(("Must provide `data` or ",
                              "`x.attrs = {'center': float, 'scale': float}"))
    else:
        center = data.mean()
        scale = data.std()
    return center + scale * x


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
    if isinstance(x, TensorSharedVariable):
        start = 0 if include_const else 1
        return pm.math.stack([x**i for i in range(start, poly_order+1)], axis=1)
    else:
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


# -----------------------------------------------------------------------------
#         Model comparison
# -----------------------------------------------------------------------------
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
    coefs = [dataset_to_series(m.coef) for m in models]
    stds = [m.std for m in models]

    def transform_ct(ct, mnames=None, params=None, value_name='coef'):
        """Make coefficient table tidy for plotting"""
        if mnames is not None:
            ct.columns = mnames
        ct.index.name = 'param'
        ct.columns.name = 'model'
        # Use params to filter by indexed variables 'a__0', 'a__1', etc.
        # should result from passing params=['a']
        if params is not None:
            try:
                subtables = [ct.loc[params]]  # track each filtered table
            except KeyError:
                subtables = []
            for p in params:
                subtables.append(ct.filter(regex=f"^{p}__[0-9]+", axis=0))
            ct = pd.concat(subtables).drop_duplicates()
        ct = (ct.T  # organize by parameter, then model
                .melt(ignore_index=False, value_name=value_name)
                .set_index('param', append=True)
                .reorder_levels(['param', 'model'])
                # .sort_index()  # do not sort to keep order of input
              )
        return ct

    ct = transform_ct(pd.concat(coefs, axis=1), mnames, params)
    if not std:
        return ct

    cs = transform_ct(pd.concat(stds, axis=1), mnames, params,
                      value_name='std')
    return pd.concat([ct, cs], axis='columns')


def plot_coef_table(ct, q=0.89, by_model=False, fignum=None):
    """Plot the table of coefficients from `sts.coef_table`.

    Parameters
    ----------
    ct : :obj:`CoefTable`
        Coefficient table output from `coef_table`.
    q : float in [0, 1], optional
        The probability interval to plot.
    by_model : bool, optional
        If True, order the coefficients by model on the y-axis, with colors to
        denote parameters; otherwise, parameters will be the y-axis, with
        colors to denote model.
    fignum : int, optional
        Figure number in which to plot the coefficients. If the figure exists,
        it will be cleared. If no figure exists, a new one will be created.

    Returns
    -------
    fig, ax : Figure and Axes where the plot was made.
    """
    fig = plt.figure(fignum, clear=True, constrained_layout=True)
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[-1]  # take most recent

    y, hue = ('model', 'param') if by_model else ('param', 'model')

    # Correct errorbars require index to be sorted.
    ct = ct.reorder_levels([hue, y]).sort_index()

    # Leverage Seaborn for basic setup
    # FIXME in seaborn 0.13.0, `dodge` ≠ False fails when there is only one
    # model to plot. "float division by zero".
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'is_categorical_dtype')
        sns.pointplot(data=ct.reset_index(), x='coef', y=y, hue=hue,
                      join=False, dodge=0.3, ax=ax)

    # FIXME get_coords is broken
    # warnings.warn('get_coords broken in Seaborn 0.13.0. No errorbars shown.')
    # Find the x,y coordinates for each point
    xc, yc, colors = get_coords(ax)

    # Manually add the errorbars since we have std values already
    z = stats.norm.ppf(1 - (1 - q)/2)
    errs = 2 * ct['std'] * z  # ± err -> 2σz
    errs = errs.dropna()
    ax.errorbar(xc, yc, fmt=' ', xerr=errs, ecolor=colors)

    # Plot the origin and move the legend outside the plot for clarity
    ax.axvline(0, ls='--', c='k', lw=1, alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    return fig, ax


# TODO make a CompareTable object?
#   * Include `sort` as a method.
#   * Print lower precision by default.
def compare(models, mnames=None, ic='WAIC', sort=False):
    """Create a comparison table of models based on information criteria.

    Parameters
    ----------
    models : list of `Quap`
        The models over which to summarize.
    mnames : list of str, optional
        Names of the models. If None, models will be numbered sequentially in
        order of input.
    ic : str in {'WAIC', 'LOOIC', 'PSIS'}
    sort : bool
        If True, sort the result by the difference in WAIC values.

    Returns
    -------
    result : dict with {'ct', 'dSE'}
        A dictionary with keys
        'ct' : pd.DataFrame
            DataFrame of the information criteria and their standard deviations.
        'dSE' : pd.DataFrame
            A symmetric matrix of the difference in standard errors of the
            pointwise information criteria of each model.
    """
    models = list(models)
    M = len(models)
    if M < 2:
        raise ValueError('Need more than one model to compare!')

    if ic not in ['WAIC', 'LOOIC', 'PSIS']:
        raise ValueError(f"Unrecognized {ic = }! Use 'WAIC' or 'LOOIC'.")

    if ic == 'LOOIC':
        ic = 'PSIS'

    if mnames is None:
        mnames = [f"m{i}" for i in range(len(models))]
    else:
        # Model names must be strings for sns.pointplot to work properly!
        mnames = [str(x) for x in list(mnames)]

    func = WAIC if ic == 'WAIC' else LOOIS
    diff_ic = f"d{ic}"

    # Create the dataframe of information criteria, with (model, var) as index
    df = (
        pd.concat(
            [pd.DataFrame(func(m)) for m in models],
            keys=mnames,
            names=['model', 'var'],
            axis='columns'
        )
        .T  # transpose for (model, var) as rows
        .drop('lppd', axis='columns')
        .sort_index()
    )

    # Subtract the minimium from each observed variable to get the diff
    df[diff_ic] = df[ic].groupby('var').transform(lambda x: x - x.min())

    # Find model with the most observed RVs
    # TODO just take intersection of all models' observed RVs. If an observed
    # RV does not exist in a model, why bother comparing?
    max_model = None
    max_obs = 0
    for m in models:
        if len(m.model.observed_RVs) > max_obs:
            max_obs = len(m.model.observed_RVs)
            max_model = m

    # Compute difference in standard error
    var_names = [v.name for v in max_model.model.observed_RVs]
    dSE = dict()
    # For each var, get column of dSE matrix corresponding to minIC model
    cf = df.unstack('var').sort_index()
    for v in var_names:
        tf = pd.DataFrame(np.nan * np.empty((M, M)),
                          index=mnames, columns=mnames)
        for i in range(M):
            for j in range(i+1, M):
                mi, mj = mnames[i], mnames[j]
                ic_i = func(models[i], pointwise=True)
                ic_j = func(models[j], pointwise=True)
                # If variable is not in both models, skip it
                try:
                    diff = ic_i[v][ic] - ic_j[v][ic]
                    tf.loc[mi, mj] = np.sqrt(len(diff) * np.var(diff))
                    tf.loc[mj, mi] = tf.loc[mi, mj]
                except KeyError:
                    continue

        dSE[v] = tf

        # Assign df['dSE'] to the appropriate column of dSE_matrix
        # NOTE could concatenate the dSE's with [var, model] index and then
        # loop over vars again to assign the dSE column and (potentially) avoid
        # this unstacking mess.
        cf[('dSE', v)] = tf[cf[(diff_ic, v)].idxmin()]

    df = cf.stack('var').reorder_levels(['var', 'model']).sort_index()

    df['weight'] = np.exp(-0.5 * df[diff_ic])
    df['weight'] /= df['weight'].groupby('var').sum()

    if sort:
        # Sort within each observed variable group
        df = df.sort_values(['var', diff_ic]).groupby('var').head(len(df))

    # Reorganize for output
    df = df[[ic, 'SE', diff_ic, 'dSE', 'penalty', 'weight']]

    return dict(ct=df, dSE_matrix=dSE)


# TODO transpose flag to match book figures with WAIC on x-axis.
def plot_compare(ct, fignum=None):
    """Plot the table of information criteria from `sts.compare`.

    Parameters
    ----------
    ct : :obj:`CompareTable`
        Compare table output from `compare`.
    fignum : int, optional
        Figure number in which to plot the coefficients. If the figure exists,
        it will be cleared. If no figure exists, a new one will be created.

    Returns
    -------
    fig, ax : Figure and Axes where the plot was made.
    """
    # TODO transpose flag (swap [xy] args and [xy]err)
    fig = plt.figure(fignum, clear=True, constrained_layout=True)
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[-1]  # take most recent

    if 'WAIC' in ct.columns:
        ic = 'WAIC'
    elif 'PSIS' in ct.columns or 'LOOIC' in ct.columns:
        ic = 'PSIS'

    # Leverage Seaborn for basic setup
    # FIXME in seaborn 0.13.0, `dodge` ≠ False fails when there is only one
    # model to plot. "float division by zero".
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'is_categorical_dtype')
        sns.pointplot(data=ct.reset_index(), y=ic, x='model', hue='var',
                      join=False, dodge=0.3, ax=ax)

    # FIXME get_coords is broken
    # warnings.warn('get_coords broken in Seaborn 0.13.0. No errorbars shown.')
    # Find the x,y coordinates for each point
    xc, yc, colors = get_coords(ax)

    # Manually add the errorbars since we have std values already
    ax.errorbar(xc, yc, yerr=ct['SE'], fmt=' ', ecolor=colors)

    # Plot in-sample deviance values
    dev_in = ct[ic] - ct['penalty']**2
    ax.scatter(xc, dev_in,
               marker='o', ec=colors, fc='none',
               label='In-Sample Deviance')

    # Plot the standard error of the *difference* in WAIC values.
    ax.errorbar(xc - 0.1, yc, yerr=ct['dSE'],
                fmt=' ', ecolor='k', lw=1, label='dSE')

    ax.axhline(ct[ic].min(), ls='--', c='k', lw=1, alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    return fig, ax


# FIXME seaborn 0.13.0 breaks this code.
# ax.collection is now empty, and ax.lines has the data we want as Line2D
# objects. See:
# <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html>
# and get_data, get_[xy]data, get_markerfacecolor, etc. methods.
def get_coords(ax):
    """Return the x, y, and color coordinates of the axes."""
    pts = [
        (x, y, point_pair.get_facecolor())
        for point_pair in ax.collections
        for x, y in point_pair.get_offsets()
        if not np.ma.is_masked(x) and not np.ma.is_masked(y)
    ]
    xc, yc, colors = [np.asarray(x) for x in zip(*pts)]
    return xc, yc, colors


# -----------------------------------------------------------------------------
#         Dataset/Frame conversion utilities
# -----------------------------------------------------------------------------
logsumexp = _logsumexp


def numpy_to_data_array(values, var_name=None):
    """Convert numpy array to an xarray.DataArray."""
    if var_name is None:
        var_name = 'data'
    dims, coords = generate_dims_coords(np.shape(values), var_name)
    return xr.DataArray(values, dims=dims, coords=coords)


def dict_to_dataset(data):
    """Convert a dictionary to an xarray.Dataset with default dimensions."""
    data_vars = {
        k: numpy_to_data_array(v, var_name=k)
        for k, v in data.items()
    }
    return xr.Dataset(data_vars=data_vars)


def flatten_dataset(ds):
    """Return a flattened DataArray and a vector of variable names."""
    return ds.to_stacked_array('data', sample_dims=[], name='data')


# TODO change to use [0], [1], etc. instead of __0, __1? This format would be
# consistent with both r::rethinking and `arviz.summary(idata)` presentation.
def cnames_from_dataset(ds):
    """Return a list of variable names from the flattened Dataset."""
    da = flatten_dataset(ds)
    # Flatten vector names into singletons 'b__0', 'b__1', ..., 'b__n'
    # TODO case of 2D, etc. variables
    df = pd.DataFrame(da.coords['variable'].values)
    g = df.groupby(0)
    df.loc[g[0].transform('size').gt(1), 0] += '__' + g.cumcount().astype(str)
    return list(df[0].values)


def dataset_to_series(ds):
    """Return a Series indexed by the data variables."""
    return pd.Series(flatten_dataset(ds), index=cnames_from_dataset(ds))


def frame_to_dataset(df, model=None):
    """Convert DataFrame to ArviZ Dataset by combinining columns with
    multi-dimensional parameters, e.g. β__0, β__1, ..., β__N into β (N,).
    """
    model = pm.modelcontext(model)
    var_names = df.columns.str.replace('__[0-9]+', '', regex=True).unique()
    the_dict = dict()
    for v in var_names:
        # Add 'chain' dimension to match expected shape
        cols = df.filter(regex=fr"^{v}(__\d+)?$")
        the_dict[v] = np.expand_dims(cols.values, 0)
    ds = az.convert_to_dataset(the_dict)
    # Remove dims for scalar variables with shape ()
    shapes = model.eval_rv_shapes()
    for v in var_names:
        if shapes[v] == ():
            ds = ds.squeeze(dim=f"{v}_dim_0", drop=True)
    return ds


# TODO Filter variable names? include/exclude?
def dataset_to_frame(ds):
    """Convert ArviZ Dataset to DataFrame by separating columns with
    multi-dimensional parameters, e.g. β (N,) into β__0, β__1, ..., β__N.

    .. note::
        This function assumes that the first dimension of a multidimensional
        DataArray is the `index` dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables with at least 1 dimension.

    Returns
    -------
    result : pandas.DataFrame
        DataFrame with columns for each variable in the dataset. Vector
        variables will be separated into columns.
    """
    if 'chain' in ds.dims:
        ds = ds.mean('chain')

    dfs = list()
    for vname, da in ds.items():
        if da.ndim == 1:
            data = da.values
            columns = [vname]
        elif da.ndim > 1:
            if da.shape[1] == 1:
                data = da.values.squeeze()
                columns = [vname]
            else:
                data = da.values
                columns = _names_from_vec(vname, da.shape[1])
        else:
            raise ValueError(f"{vname} has invalid dimension {da.ndim}.")

        dfs.append(pd.DataFrame(data=data, columns=columns))

    df = pd.concat(dfs, axis=1)
    df.index.name = da.dims[0]
    return df


def _names_from_vec(vname, ncols):
    """Create a list of strings ['x__0', 'x__1', ..., 'x__``ncols``'],
    where 'x' is ``vname``."""
    # TODO case of 2D, etc. variables
    fmt = '02d' if ncols > 10 else 'd'
    return [f"{vname}__{i:{fmt}}" for i in range(ncols)]


# -----------------------------------------------------------------------------
#         Information Criteria Functions
# -----------------------------------------------------------------------------
# TODO
# * use constant string for inference_data and loglikelihood docs.
# * use constant string for lppd, WAIC, LOOIS, LOOCV docs.

def inference_data(model, post=None, var_names=None, eval_at=None, Ns=1000):
    """Prepare the inference data structure for the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used. Note that the posterior
        distribution is *not* recomputed (_i.e._ the model is not re-fit to the
        ``eval_at`` data). This behavior allows the user to train the model
        with one dataset, and test it with another.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : az.InferenceData
        An InferenceData object with groups
        posterior : xarray.Dataset of (chain=1, draw=Ns, var_dim)
            The posterior distribution for each unobserved model parameter.
        log_likelihood : xarray.Dataset of (chain=1, draw=Ns, var_dim=N)
            The log likelihood of each observed variable.
    """
    if post is None:
        Ns = int(Ns)
        post = model.get_samples(Ns)

    if 'chain' not in post.dims:
        post = post.expand_dims('chain', axis=0)

    if eval_at is not None:
        for k, v in eval_at.items():
            model.model.set_data(k, v)

    return pm.compute_log_likelihood(
        idata=az.convert_to_inference_data(post),
        model=model.model,
        var_names=var_names,
        progressbar=False,
    )


def loglikelihood(model, post=None, var_names=None, eval_at=None, Ns=1000):
    """Compute the log-likelihood of the data, given the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``quap`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : xarray.Dataset (draw=Ns, var_dim=N)
        Log likelihood for each of the ``var_names``. Each will be an array of
        size (Ns, N), for ``Ns`` samples of the posterior, and `N` data points.
    """
    return (
        inference_data(model, post, var_names, eval_at, Ns)
        .log_likelihood
        .mean('chain')
    )


def deviance(model=None, loglik=None, post=None, var_names=None, eval_at=None,
             Ns=1000):
    """Compute the deviance as -2 * lppd."""
    the_lppd = lppd(
        model=model,
        loglik=loglik,
        post=post,
        var_names=var_names,
        eval_at=eval_at,
        Ns=Ns,
    )
    return {k: -2 * v.sum() for k, v in the_lppd.items()}


# TODO combine identical documentation from lppd, WAIC, LOOIS functions.
def lppd(model=None, loglik=None, post=None, var_names=None, eval_at=None,
         Ns=1000):
    r"""Compute the log pointwise predictive density for a model.

    The lppd is defined as follows:

    .. math::
        \text{lppd}(y, \Theta) = \sum_i \log \frac{1}{S} \sum_s p(y_i | \Theta_s)

    where :math:`S` is the number of samples, and :math:`\Theta_s` is the
    :math:`s`-th set of sampled parameter values in the posterior distribution.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object. If ``loglik`` is not given, ``model`` will be
        used to compute it.
    loglik : dict like {var_name: (Ns, N) ndarray}
        The log-likelihood of each desired output variable in the model, where
        `Ns` is the number of samples of the posterior, and `N` is the number
        of data points for that variable.

    .. note::
        Only one of ``model`` or ``loglik`` may be given.

    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``model`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : dict like {var_name: lppd}
        A dictionary of the lppd for each observed variable.
    """
    if model is None and loglik is None:
        raise ValueError('One of `model` or `loglik` must be given!')

    if loglik is None:
        loglik = loglikelihood(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns,
        )

    if Ns is None:
        Ns = loglik.sizes['draw']

    if var_names is None:
        var_names = loglik.keys()

    return {v: logsumexp(loglik[v], axis=0) - np.log(Ns) for v in var_names}


def DIC(model, post=None, Ns=1000):
    """Compute the Deviance Information Criteria of the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``model`` distribution.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : dict with keys {'dic', 'pD'}
        'dic' : float
            The Deviance Information Criteria
        'pD' : float
            The penalty term.

    References
    ----------
    [1]: Gelman (2020). Bayesian Data Analysis, 3 ed. pp 172--173.
    """
    if post is None:
        post = model.get_samples(Ns)
    f_loglik = model.model.compile_logp()
    dev = [-2 * f_loglik(post.iloc[i]) for i in range(Ns)]
    dev_hat = model.deviance
    pD = dev.mean() - dev_hat
    return dict({'dic': dev_hat + 2*pD, 'pD': pD})


def WAIC(model=None, loglik=None, post=None, var_names=None, eval_at=None,
         Ns=1000, pointwise=False):
    r"""Compute the Widely Applicable Information Criteria for the model.

    The WAIC is defined as:

    .. math::
        \text{WAIC}(y, \Theta) = -2 \left(
            \text{lppd}
            - \sum_i \var_\Theta \log p(y_i | \Theta)
        \right)

    An estimate of the standard error for out-of-sample deviance is:

    .. math::
        s_{\text{WAIC}} = \sqrt{ N \var\[-2 (\text{lppd}_i - p_i) \] }

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object. If ``loglik`` is not given, ``model`` will be
        used to compute it.
    loglik : dict like {var_name: (Ns, N) ndarray}
        The log-likelihood of each desired output variable in the model, where
        `Ns` is the number of samples of the posterior, and `N` is the number
        of data points for that variable.

    .. note::
        Only one of ``model`` or ``loglik`` may be given.

    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``model`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.
    pointwise : bool
        If True, return a vector of length `N` for each output variable, where
        `N` is the number of data points for that variable.

    Returns
    -------
    result : dict like {var_name: WAIC}
        A dictionary of the WAIC for each observed variable.
    """
    if model is None and loglik is None:
        raise ValueError('One of `model` or `loglik` must be given!')

    if loglik is None:
        loglik = loglikelihood(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns,
        )

    the_lppd = lppd(loglik=loglik, var_names=var_names)

    out = dict()
    for v in the_lppd:
        penalty = loglik[v].var(dim='draw').values
        waic_vec = -2 * (the_lppd[v] - penalty)
        n_cases = loglik[v].shape[1]
        std_err = (n_cases * np.var(waic_vec))**0.5

        if pointwise:
            lppd_w, w, p = the_lppd[v], waic_vec, penalty
        else:
            lppd_w, w, p = the_lppd[v].sum(), waic_vec.sum(), penalty.sum()

        d = dict(WAIC=w, lppd=lppd_w, penalty=p, SE=std_err)
        out[v] = pd.DataFrame(d) if pointwise else pd.Series(d)

    return out


def LOOIS(model=None, idata=None, post=None, var_names=None, eval_at=None,
          Ns=1000, pointwise=False):
    """Compute the Pareto-smoothed Importance Sampling Leave-One-Out
    Cross-Validation score of the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    post : (Ns, p) DataFrame
        Samples of the posterior distribution with model free variables as
        column names. If ``post`` is not given, ``Ns`` samples will be drawn
        from the ``model`` distribution.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.
    pointwise : bool
        If True, return a vector of length `N` for each output variable, where
        `N` is the number of data points for that variable.

    Returns
    -------
    result : dict like {var_name: LOOIS}
        A dictionary of the LOOIS for each observed variable.
    """
    if model is None and idata is None:
        raise ValueError('One of `model` or `idata` must be given!')

    if idata is None:
        idata = inference_data(
            model=model,
            post=post,
            var_names=var_names,
            eval_at=eval_at,
            Ns=Ns
        )

    if var_names is None:
        var_names = idata.log_likelihood.data_vars

    out = dict()
    for v in var_names:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            loo = az.loo(idata, pointwise=pointwise, var_name=v)

        elpd = loo.loo_i if pointwise else loo.elpd_loo
        d = dict(
            PSIS=-2*elpd,       # == loo_list$estimates['looic', 'Estimate']
            lppd=elpd,
            penalty=loo.p_loo,  # == loo_list$p_loo
            SE=2*loo.se,        # == loo_list$estimates['looic', 'SE']
        )
        out[v] = pd.DataFrame(d) if pointwise else pd.Series(d)

    return out


# Alias
PSIS = LOOIS


# "Globalize" nested function for parallelization.
# See <https://gist.github.com/EdwinChan/3c13d3a746bb3ec5082f>
@contextmanager
def globalized(func):
    namespace = sys.modules[func.__module__]
    name, qualname = func.__name__, func.__qualname__
    func.__name__ = func.__qualname__ = f"__{name}_{uuid.uuid4().hex}"
    setattr(namespace, func.__name__, func)
    try:
        yield
    finally:
        delattr(namespace, func.__name__)
        func.__name__, func.__qualname__ = name, qualname


def LOOCV(model, ind_var, obs_var, out_var, X_data, y_data,
          lno=1, pointwise=False, parallel=True, progressbar=True):
    """Compute the leave-one-out cross-validation score of the model.

    Parameters
    ----------
    model : :obj:`Quap`
        The fitted model object.
    ind_var : str
        The name of the independent data variable in the model.
    obs_var : str
        The name of the observed data variable in the model.
    out_var : str
        The name of the output variable in the model.
    X_data : array_like
        The input data.
    y_data : array_like
        The output data.
    lno : int
        Number of data points to leave out per iteration.
    pointwise : bool
        If True, return the score for each data point.
    parallel : bool
        If True, run the main loop in parallel.
    progressbar : bool
        If True, print a progress bar.

    Returns
    -------
    result : dict
        Dictionary of results with attributes:
        loocv : float or (N,) array
            The deviance-scaled score.
        lppd : float or (N,) array
            The least pointwise posterior density of the data.
        std : float
            An estimate of the standard error of the ``loocv`` score.
    """
    N = len(y_data)
    M = N // lno  # number of chunks of data

    X_list = np.split(X_data, M)
    y_list = np.split(y_data, M)

    def leave_out(data_list, i):
        """Return an array with one element of the list removed."""
        return np.concatenate([data_list[j] for j in range(M) if j != i])

    # Fit a model to each chunk of data
    def loocv_func(i):
        """Perform cross-validation on data chunk `i`."""
        # Train the model on the data without chunk i
        model.model.set_data(ind_var, leave_out(X_list, i))
        model.model.set_data(obs_var, leave_out(y_list, i))
        test_quap = quap(model=model.model)

        # Compute the LPPD on the left-out chunk of data
        return (
            lppd(
                model=test_quap,
                eval_at={ind_var: X_list[i], obs_var: y_list[i]}
            )
            [out_var]
        )

    # NOTE **WARNING** SLOW CODE
    if parallel:
        with globalized(loocv_func):
            if progressbar:
                lppd_list = process_map(
                    loocv_func,
                    range(M),
                    desc='LOOCV',
                    leave=False,
                    max_workers=4,
                    position=1,
                )
            else:
                with Pool(4) as pool:
                    pool.map(loocv_func, range(M))
    else:
        iters = range(M)
        if progressbar:
            from tqdm import tqdm
            iters = tqdm(iters, desc='LOOCV')
        lppd_list = [loocv_func(i) for i in iters]

    lppd_cv = np.array(lppd_list).squeeze()
    c = lppd_cv if pointwise else lppd_cv.sum()

    # mean of chunks, var of data
    var = lppd_cv.var() if lno == 1 else lppd_cv.mean(axis=0).var()
    std_err = (N * var)**0.5

    # TODO return DataFrame if pointwise, else Series
    return dict(
        loocv=-2*c,
        lppd=c,
        std=std_err,
    )


# -----------------------------------------------------------------------------
#         Simulations
# -----------------------------------------------------------------------------
# (R code 7.17 - 7.19)
def sim_train_test(
    N=20,
    k=3,
    rho=np.r_[0.15, -0.4],
    b_sigma=100,
    compute_WAIC=False,
    compute_LOOIC=False,
    compute_LOOCV=False
):
    r"""Simulate fitting a model of `k` parameters to `N` data points.

    The default data-generating (i.e. "true") model used is:

    .. math::
            y_i ~ \mathcal{N}(μ_i, 1)
            μ_i = α + β_1 x_{1,i} + β_2 x_{2,i}
            α   = 0
            β_1 =  0.15
            β_2 = -0.4

    If `k` is greater than 3, additional :math:`x_{j,i}` and corresponding
    :math:`\beta_j` values will be included in the model, but should have no
    effect on predictive power since the underlying data does not depend on any
    additional parameters.

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
    compute_WAIC, compute_LOOIC, compute_LOOCV : bool
        If True, compute the WAIC, LOOIC, and/or LOOCV criteria, respectively.

    Returns
    -------
    result : dict
        'dev' : dict with keys {'train', 'test'}
            The deviance of the model for the train and test data.
        'model' : :obj:Quap
            The model itself.
    """
    Y_SIGMA = 1
    # NOTE this line is *required* for expected parallel behavior
    np.random.seed()

    # Define the dimensions of the "true" distribution to match the model
    n_dim = 1 + len(rho)
    if n_dim < k:
        n_dim = k

    # Generate train/test data
    # NOTE this method of creating the data obfuscates the underlying linear
    # model, but it is a clean way of accommodating varying parameter lengths.
    Rho = np.eye(n_dim)
    Rho[0, 1:len(rho)+1] = rho
    Rho[1:len(rho)+1, 0] = rho

    # >>> Rho
    # === array([[ 1.  ,  0.15, -0.4 , 0. ,      0. ],
    #            [ 0.15,  1.  ,  0.  , 0. ,      0. ],
    #            [-0.4 ,  0.  ,  1.  , 0. ,      0. ],
    #            [ 0.  ,  0.  ,  0.  , 1. ,      0. ],
    #              ...          ...       , ...
    #            [ 0.  ,  0.  ,  0.  , 0. ,      1. ]])
    #

    true_dist = stats.multivariate_normal(mean=np.zeros(n_dim), cov=Rho)
    X_train = true_dist.rvs(N)  # (N, k)
    X_test = true_dist.rvs(N)

    # Separate the inputs and outputs for readability
    y_train, X_train = X_train[:, 0], X_train[:, 1:]
    y_test, X_test = X_test[:, 0], X_test[:, 1:]

    # Define the training matrix
    mm_train = np.ones((N, 1))  # intercept term
    if k > 1:
        mm_train = np.c_[mm_train, X_train[:, :k-1]]

    # Build and fit the model to the training data
    with pm.Model():
        X = pm.MutableData('X', mm_train)
        obs = pm.MutableData('obs', y_train)
        α = pm.Normal('α', 0, 100, shape=(1,))  # flat prior always
        if k == 1:
            μ = pm.Deterministic('μ', α)
        else:
            βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
            β = pm.math.concatenate([α, βn])
            μ = pm.Deterministic('μ', pm.math.dot(X, β))
        y = pm.Normal('y', μ, Y_SIGMA, observed=obs, shape=obs.shape)
        q = quap(data=dict(X=mm_train, obs=y_train))

    # -------------------------------------------------------------------------
    #         Compute the Information Criteria
    # -------------------------------------------------------------------------
    # NOTE WAIC, LOOIC, LOOCV must be computed on the *TRAINING* data, not the
    # test data. The idea is that these criteria are used to predict what the
    # test error would be without having the test data available.
    # Store these values for WAIC and LOOIC:
    idata_train = inference_data(q)
    loglik_train = idata_train.log_likelihood.mean('chain')

    lppd_train = lppd(loglik=loglik_train)['y']

    # Compute the lppd with the test data
    mm_test = np.ones((N, 1))
    if k > 1:
        mm_test = np.c_[mm_test, X_test[:, :k-1]]

    # NOTE inference_data has the side effect of setting the model data to
    # `eval_at`, and not changing it back. No known way to extract the data
    # from the model (e.g. within a function call) without explicitly providing
    # the original data variable names and calling `q.model.X.eval()`. Can also
    # rely on user to store `q.data` as a df/dict with the proper names.

    # Compute the posterior and log-likelihood of the test data
    lppd_test = lppd(q, eval_at={'X': mm_test, 'obs': y_test})['y']

    # Compute the deviance
    res = pd.Series({('deviance', 'train'): -2 * np.sum(lppd_train),
                     ('deviance',  'test'): -2 * np.sum(lppd_test)})

    # Compile Results
    if compute_WAIC:
        res[('WAIC', 'test')] = WAIC(loglik=loglik_train)['y']['WAIC']

    if compute_LOOIC:
        res[('LOOIC', 'test')] = LOOIS(idata=idata_train)['y']['PSIS']

    if compute_LOOCV:
        res[('LOOCV', 'test')] = LOOCV(
            model=q,
            ind_var='X',
            obs_var='obs',
            out_var='y',
            X_data=mm_train,
            y_data=y_train,
        )['loocv']

    return dict(res=res, model=q)


# =============================================================================
# =============================================================================
