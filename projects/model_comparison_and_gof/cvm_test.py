import numpy as np
import scipy.stats
from scipy.optimize import shgo
    
def cramervonmises_2samp_weighted(data1, data2, weights1=None, weights2=None, method='auto',):
    
    supported_types = [np.ndarray, pd.Series]
    #Here it tries to set weigths=1 when user does not pass them
    #Likely to break if input is not np.array
    if not type(weights1) in supported_types:
        if weights1==None:
            weights1 = np.ones_like(data1)
            
    if not type(weights2) in supported_types:
        if weights2==None:
            weights2 = np.ones_like(data2)

    xa = np.sort(np.asarray(data1))
    ya = np.sort(np.asarray(data2))
    
    weights1_n = weights1 / np.sum(weights1) * 1.
    weights2_n = weights2 / np.sum(weights2) * 1.

    if xa.size <= 1 or ya.size <= 1:
        raise ValueError('x and y must contain at least two observations.')
    if xa.ndim > 1 or ya.ndim > 1:
        raise ValueError('The samples must be one-dimensional.')
    if method not in ['auto', 'exact', 'asymptotic']:
        raise ValueError('method must be either auto, exact or asymptotic.')

    nx = len(xa)
    ny = len(ya)

    if method == 'auto':
        if max(nx, ny) > 10:
            method = 'asymptotic'
        else:
            method = 'exact'

    # get ranks of x and y in the pooled sample
    z = np.concatenate([xa, ya])
    x_uniq = np.unique(np.concatenate([xa, ya]))
    
    inds1 = np.searchsorted(x_uniq, data1)
    inds2 = np.searchsorted(x_uniq, data2)
    # in case of ties, use midrank (see [1])
    r = scipy.stats.rankdata(z, method='average')
    rx = r[:nx]
    ry = r[nx:]

    # compute U (eq. 10 in [2])
    u = nx * np.sum((rx - np.arange(1, nx+1))**2)
    u += ny * np.sum((ry - np.arange(1, ny+1))**2)

    # compute T (eq. 9 in [2])
    k, N = nx*ny, nx + ny
    t = u / (k*N) - (4*k - 1)/(6*N)

    if method == 'exact':
        p = _pval_cvm_2samp_exact(u, nx, ny)
    else:
        # compute expected value and variance of T (eq. 11 and 14 in [2])
        et = (1 + 1/N)/6
        vt = (N+1) * (4*k*N - 3*(nx**2 + ny**2) - 2*k)
        vt = vt / (45 * N**2 * 4 * k)

        # computed the normalized statistic (eq. 15 in [2])
        tn = 1/6 + (t - et) / np.sqrt(45 * vt)

        # approximate distribution of tn with limiting distribution
        # of the one-sample test statistic
        # if tn < 0.003, the _cdf_cvm_inf(tn) < 1.28*1e-18, return 1.0 directly
        if tn < 0.003:
            p = 1.0
        else:
            p = max(0, 1. - _cdf_cvm_inf(tn))

    #return CramerVonMisesResult(statistic=t, pvalue=p)
    return (t, p)


# From https://github.com/scipy/scipy/blob/main/scipy/stats/_hypotests.py
def _cdf_cvm_inf(x):
    """
    Calculate the cdf of the Cramér-von Mises statistic (infinite sample size).
    See equation 1.2 in Csörgő, S. and Faraway, J. (1996).
    Implementation based on MAPLE code of Julian Faraway and R code of the
    function pCvM in the package goftest (v1.1.1), permission granted
    by Adrian Baddeley. Main difference in the implementation: the code
    here keeps adding terms of the series until the terms are small enough.
    The function is not expected to be accurate for large values of x, say
    x > 4, when the cdf is very close to 1.
    """
    x = np.asarray(x)

    def term(x, k):
        # this expression can be found in [2], second line of (1.3)
        u = np.exp(gammaln(k + 0.5) - gammaln(k+1)) / (np.pi**1.5 * np.sqrt(x))
        y = 4*k + 1
        q = y**2 / (16*x)
        b = kv(0.25, q)
        return u * np.sqrt(y) * np.exp(-q) * b

    tot = np.zeros_like(x, dtype='float')
    cond = np.ones_like(x, dtype='bool')
    k = 0
    while np.any(cond):
        z = term(x[cond], k)
        tot[cond] = tot[cond] + z
        cond[cond] = np.abs(z) >= 1e-7
        k += 1

    return tot


def _pval_cvm_2samp_exact(s, nx, ny):
    """
    Compute the exact p-value of the Cramer-von Mises two-sample test
    for a given value s (float) of the test statistic by enumerating
    all possible combinations. nx and ny are the sizes of the samples.
    """
    rangex = np.arange(nx)
    rangey = np.arange(ny)

    us = []

    # x and y are all possible partitions of ranks from 0 to nx + ny - 1
    # into two sets of length nx and ny
    # Here, ranks are from 0 to nx + ny - 1 instead of 1 to nx + ny, but
    # this does not change the value of the statistic.
    for x, y in _all_partitions(nx, ny):
        # compute the statistic
        u = nx * np.sum((x - rangex)**2)
        u += ny * np.sum((y - rangey)**2)
        us.append(u)

    # compute the values of u and the frequencies
    u, cnt = np.unique(us, return_counts=True)
    return np.sum(cnt[u >= s]) / np.sum(cnt)


def _all_partitions(nx, ny):
    """
    Partition a set of indices into two fixed-length sets in all possible ways
    Partition a set of indices 0 ... nx + ny - 1 into two sets of length nx and
    ny in all possible ways (ignoring order of elements).
    """
    z = np.arange(nx+ny)
    for c in combinations(z, nx):
        x = np.array(c)
        mask = np.ones(nx+ny, bool)
        mask[x] = False
        y = z[mask]
        yield x, y