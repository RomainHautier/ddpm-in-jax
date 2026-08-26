"""Shared look for all plot scripts. Edit here to restyle every figure at once."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#232823'
GT_COLOR, GT_LW = '#000000', 3.4      # ground truth: always thickest and pure black
PALETTE = ['#1f77b4', '#2a78d6', '#7a4bd0', '#0f9e78', '#d4770a', '#b8399e', '#c22f4f',
           '#8a5cd6', '#28658a', '#9aa198']
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
BAND_LABELS = ['[1,5)', '[5,16)', '[16,32)', '[32,64)', '[64,96)']
SHADE_MID, SHADE_HI = '#f8e8d0', '#e3edf3'          # [16,32) and [32,96) band shading

# human-readable names for keys used in the result stores
STRATEGY = {'none': 'unguided', 'residual': 'residual dial', 'reward': 'dose dial v1',
            'placement': 'placement dial', 'all3': 'all three (v1)',
            'rewardv2': 'dose dial v2 (mid-band)', 'all3v2': 'all three (v2)'}
MODEL = {'base0': 'base', 'r1k-449': 'fine-tuned Re=1000', 're2k-149': 'fine-tuned Re=2000',
         'pr2k-549': 'placement-reward Re=2000', 'pr1k-549': 'placement-reward Re=1000',
         'rs8kkl-799': 'fine-tuned Re=8000 (KL)', 'st8k-599': 'steered-trained Re=8000',
         'st2k-599': 'steered-trained Re=2000', 'st1k-599': 'steered-trained Re=1000'}


def apply(fontsize=10, dpi=140):
    plt.rcParams.update({'font.size': fontsize, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.dpi': dpi, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                         'legend.frameon': False})


def shade_bands(ax):
    ax.axvspan(16, 32, color=SHADE_MID, zorder=0)
    ax.axvspan(32, 96, color=SHADE_HI, zorder=0)


def re_axis(ax, regs=(1000, 2000, 4000, 8000)):
    ax.set_xscale('log'); ax.set_xticks(list(regs))
    ax.set_xticklabels([f'{r // 1000}k' for r in regs]); ax.minorticks_off()
