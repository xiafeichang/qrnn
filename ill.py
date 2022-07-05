import numpy as np
import matplotlib
matplotlib.use('cairo')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Helvetica'
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches

xboxcolor = 'seagreen'
yboxcolor = 'slateblue'
ycboxcolor = 'chocolate'

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))

ax1.plot(np.full(100,0.8), np.linspace(0.0, 0.85, 100), color='grey', linestyle='dashed', linewidth='5')
ax1.plot([0.05, 0.05, 0.05], [0.15, 0.20, 0.25], '.', color='k')
ax1.plot([0.40, 0.45, 0.50], [0.00, 0.00, 0.00], '.', color='k')
ax1.plot([0.95, 0.95, 0.95], [0.15, 0.20, 0.25], '.', color='k')

ax1.text(0.55, 0.95, r'\textbf{DATA}', color='y', fontsize=28)
ax1.text(0., 0.95, r'\textbf{\textit{Input variables}}', fontsize=20, ha='left')
ax1.text(1., 0.95, r'\textbf{\textit{Target}}', fontsize=20, ha='right')

ax1.text(0.05, 0.8, r'$X_i = (p_T,\eta,\phi,\rho)_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.05, 0.6, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.05, 0.4, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.05, 0.0, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')

ax1.text(0.95, 0.8, r'$y_1$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax1.text(0.95, 0.6, r'$y_2$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax1.text(0.95, 0.4, r'$y_3$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax1.text(0.95, 0.0, r'$y_n$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')

ax1.text(0.25, 0.6, r'$y_1$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.25, 0.4, r'$y_1$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.45, 0.4, r'$y_2$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.25, 0.0, r'$y_1$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='left')
ax1.text(0.65, 0.0, r'$y_n$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='left')

style = "Simple, tail_width=3.5, head_width=8., head_length=16."
kw = dict(arrowstyle=style, color=yboxcolor)
ax1.add_patch(patches.FancyArrowPatch((0.85, 0.83), (0.35, 0.63), connectionstyle="arc3,rad=-.2", **kw))
ax1.add_patch(patches.FancyArrowPatch((0.85, 0.63), (0.55, 0.43), connectionstyle="arc3,rad=-.1", **kw))


ax2.plot(np.full(100,0.8), np.linspace(0., 0.85, 100), color='grey', linestyle='dashed', linewidth='5')
ax2.plot([0.05, 0.05, 0.05], [0.15, 0.20, 0.25], '.', color='k')
ax2.plot([0.42, 0.47, 0.52], [0.00, 0.00, 0.00], '.', color='k')
ax2.plot([0.95, 0.95, 0.95], [0.15, 0.20, 0.25], '.', color='k')

ax2.text(0.55, 0.95, r'\textbf{MC}', color='y', fontsize=28)
ax2.text(0., 0.95, r'\textbf{\textit{Input variables}}', fontsize=20, ha='left')
ax2.text(1., 0.95, r'\textbf{\textit{Target}}', fontsize=20, ha='right')

ax2.text(0.05, 0.8, r'$X_i = (p_T,\eta,\phi,\rho)_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.05, 0.6, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.05, 0.4, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.05, 0.0, r'$X_i$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=xboxcolor, fill=False, lw=2.5), ha='left')

ax2.text(0.95, 0.8, r'$y_1$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax2.text(0.95, 0.6, r'$y_2$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax2.text(0.95, 0.4, r'$y_3$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')
ax2.text(0.95, 0.0, r'$y_n$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=yboxcolor, fill=False, lw=2.5), ha='right')

ax2.text(0.24, 0.6, r'$y_1^{corr}$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=ycboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.24, 0.4, r'$y_1^{corr}$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=ycboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.46, 0.4, r'$y_2^{corr}$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=ycboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.24, 0.0, r'$y_1^{corr}$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=ycboxcolor, fill=False, lw=2.5), ha='left')
ax2.text(0.65, 0.0, r'$y_n^{corr}$', fontsize=20, bbox=dict(boxstyle='square,pad=1', ec=ycboxcolor, fill=False, lw=2.5), ha='left')

style = "Simple, tail_width=3.5, head_width=8., head_length=16."
kw = dict(arrowstyle=style, color=ycboxcolor)
ax2.add_patch(patches.FancyArrowPatch((0.85, 0.83), (0.39, 0.63), connectionstyle="arc3,rad=-.2", **kw))
ax2.add_patch(patches.FancyArrowPatch((0.85, 0.63), (0.61, 0.43), connectionstyle="arc3,rad=-.1", **kw))


ax1.axis('off')
ax2.axis('off')

fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.85, wspace=0.2)

fig.savefig('chainedQRill.pdf')
fig.savefig('chainedQRill.png')


