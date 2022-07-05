import pickle
import gzip
import numpy as np
from joblib import delayed, Parallel

def compute_qweights(sr, qs, weights=None):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    qweights = 1./np.average(loss, axis=0, weights=weights)
    return qweights/np.min(qweights)

def Hubber(e, delta=0.1, signed=False):
    is_small_e = np.abs(e) < delta
    small_e = np.square(e) / (2.*delta)
    big_e = np.abs(e) - delta/2.
    if signed:
        return np.sign(e)*np.where(is_small_e, small_e, big_e) 
    else: 
        return np.where(is_small_e, small_e, big_e)
 
def parallelize(func, X, Y, *args, n_jobs=10, **kwargs): 
    if len(Y.shape) == 1: 
        Y = np.array(Y).reshape(-1,1)
    nY = Y.shape[-1]
    Z = np.hstack([X,Y])
    if nY > 1: 
        return np.concatenate(Parallel(n_jobs=n_jobs, verbose=20)(delayed(func)(sli[:,:-nY], sli[:,-nY:], *args, **kwargs) for sli in np.array_split(Z, n_jobs)))
    else: 
        return np.concatenate(Parallel(n_jobs=n_jobs, verbose=20)(delayed(func)(sli[:,:-1], sli[:,-1], *args, **kwargs) for sli in np.array_split(Z, n_jobs)))

def load_clf(clf_name): 
    clf = pickle.load(gzip.open(clf_name))
    return clf['clf']

def sec2HMS(sec):
    d = 0
    h = 0

    m = sec // 60
    s = sec % 60
    if m > 59: 
        h = m // 60
        m = m % 60
        if h > 23:
            d = h // 24
            h = h % 24

    return int(d), int(h), int(m), s

def mathSciNot(x, lim=(-2,3), dig=2):
    exp = int(np.log10(x))
    if exp > lim[0] and exp <= lim[1]: 
        outstr = '{:.' + str(dig) + 'f}'
        return _shorterStr(outstr.format(x), '{:g}'.format(x))
    else:
        quo = x / pow(10., exp)
        if quo > 5. :
            quo /= 10.
            exp += 1
        outstr = '{:.' + str(dig) + 'f}'
        return _shorterStr(outstr.format(quo), '{:g}'.format(quo)) + r' $\times$ 10$^{'+str(exp)+'}$'

def _shorterStr(s1,s2):
    return s1 if len(s1) < len(s2) else s2

def weighted_quantiles(a, q, weights=None, **kwargs): 
    if weights is None: 
        return np.quantile(a, q, **kwargs)

    a = np.asarray(a)
    weights = np.asarray(weights)
    idx = np.argsort(a)
    a = a[idx]
    weights = weights[idx]
    quantiles = np.cumsum(weights) / np.sum(weights)
    return np.interp(q, quantiles, a)


