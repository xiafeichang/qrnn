import numpy as np
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt

def main():

    corr = 
    names = 

    fig, ax = plt.subplots(tight_layout=True)
    im = ax.imshow(corr)
    ax.set_xticks(np.arange(len(names)), labels=names)
    ax.set_yticks(np.arange(len(names)), labels=names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(names)): 
        for j in range(len(names)): 
            ax.text(j, i, corr[i,j], ha='center', va='center', color='k')

    ax.set_title('Correlation between variables')
    plt.show()


if __name__ == '__main__': 
    main()


