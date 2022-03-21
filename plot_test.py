import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


np.random.seed(100)

n = 500000
mu = np.random.normal(0., 2., n)
sigma = np.random.uniform(1., 10., n)

var = np.random.normal(mu, sigma)

df = pd.DataFrame({'mu':mu, 'sigma':sigma, 'var':var})

n_mu_bin = 5
mu_bin_edges = np.quantile(mu, np.arange(1./n_mu_bin, 1., 1./n_mu_bin))


qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])

fig, axs = plt.subplots(2, 3,figsize=(16, 9),tight_layout=True)
fig.suptitle('plot_test')
for i, ax in zip(range(n_mu_bin), axs.flat):
    if i == 0: 
        query_str = 'mu < ' + str(mu_bin_edges[0])
    elif i < n_mu_bin - 1:
        query_str = 'mu > ' + str(mu_bin_edges[i-1]) + ' and mu < ' + str(mu_bin_edges[i])
    else:
        query_str = 'mu > ' + str(mu_bin_edges[-1])
    
    df_eval = df.query(query_str)
    quantiles = np.array([stats.norm.ppf(q, df_eval['mu'], df_eval['sigma']) for q in qs]).T
    
    ax.hist(df_eval['var'], bins=100, density=True, cumulative=True, histtype='step')
    ax.errorbar(np.mean(quantiles, axis=0), qs, xerr=np.std(quantiles, axis=0), fmt='.', markersize=7, elinewidth=2, capsize=3, markeredgewidth=2)
    ax.set_title('$\mu$ bin {}'.format(i))

plt.show()
fig.savefig('plot_test.png')
