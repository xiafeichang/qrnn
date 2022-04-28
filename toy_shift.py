import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy import stats


def compute_proba(X, scale, bias=0., noise=0.):
    seeds1 = stats.expon.ppf(X, scale=scale)
    proba_peak = stats.norm.cdf(0.2, seeds1, np.sqrt(seeds1)) + np.random.normal(bias, noise, len(X))
    return proba_peak, 1-proba_peak

def shiftYev(Y, drats, pPeak_mc, pTail_mc, pPeak_data, pTail_data):

    r=np.random.uniform()

    if Y == 0. and pTail_data>pTail_mc and r<drats[0]:
        p2t = 1
        Y_corr = np.random.exponential(3.)+0.02
    elif Y > 0. and pPeak_data>pPeak_mc and r<drats[1]:
        p2t = -1
        Y_corr = 0.
    else:
        p2t = 0
        Y_corr = Y

    return Y_corr, p2t

def get_diffrats(pPeak_mc, pTail_mc, pPeak_data, pTail_data):
    return [np.divide(pTail_data - pTail_mc,pPeak_mc), np.divide(pPeak_data - pPeak_mc,pTail_mc)]



#np.random.seed(100)

X_data = np.random.uniform(0., 1., 100000)
X_mc = np.random.uniform(0., 1., 100000)

scale_data = 3.
scale_mc = 2.9
seeds1_data = stats.expon.ppf(X_data,scale=scale_data)
seeds1_mc = stats.expon.ppf(X_mc,scale=scale_mc)

seeds2_data = np.random.normal(seeds1_data, np.sqrt(seeds1_data))
seeds2_mc = np.random.normal(seeds1_mc, np.sqrt(seeds1_mc))

data = np.apply_along_axis(lambda x: x if x>0.2 else 0., 0, seeds2_data.reshape(1,-1)).ravel()
mc = np.apply_along_axis(lambda x: x if x>0.2 else 0., 0, seeds2_mc.reshape(1,-1)).ravel()

# now shift mc, i.e. treat the arrays "mus_mc" and "mc" as "X" and "Y"
pPeak_mc, pTail_mc = compute_proba(X_mc, scale_mc, -0.01, 0.01)
pPeak_data, pTail_data = compute_proba(X_mc, scale_data, 0., 0.01) # compute the probability assuming that the data set (X, Y) is data


p2t = 0
t2p = 0
mc_shift = np.array([]) 
for i in range(len(mc)): 
    drats = get_diffrats(pPeak_mc[i], pTail_mc[i], pPeak_data[i], pTail_data[i])
    Y_shift, move = shiftYev(mc[i], drats, pPeak_mc[i], pTail_mc[i], pPeak_data[i], pTail_data[i])
    mc_shift = np.append(mc_shift, Y_shift)
    if move == 1: 
        p2t += 1
    elif move == -1:
        t2p += 1

print(f'peak to tail: {p2t}, tail to peak: {t2p}')

data_n, bin_edges = np.histogram(data, range=(0., 10.), bins=100)
x = np.array([])
xerr = np.array([])
for i in range(len(data_n)):
    x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
    xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)
data_nerr = np.sqrt(data_n)
    
fig = plt.figure(tight_layout=True)
plt.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')
plt.hist(mc, range=(0., 10.), bins=100, histtype='step', color='red', label='mc unshifted')
plt.hist(mc_shift, range=(0., 10.), bins=100, histtype='step', color='green', label='mc shifted')
plt.title('toy model')
plt.legend()
fig.savefig('plots/other_test/test_shift_toy.png')
plt.close(fig)


