"""Shared look for all plot scripts. Edit here to restyle every figure at once."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#232823'
GT_COLOR, GT_LW = '#000000', 2.0      # ground truth: pure black, and thicker than the model
                                      # curves (1.7) but no more - a heavy line masks exactly
                                      # the small deviations a spectrum plot exists to show
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


# Matched to the reference-image scripts in plotting/ (plot_lr_preprocessing, plot_frames_by_model,
# plot_steered_vs_unguided), which all use serif / Computer Modern mathtext at base size 11, so
# figures from both families sit together in the same document without a font or size mismatch.
BASE_FS = 11        # body text: axis labels, ticks
TITLE_FS = 12       # panel titles
SUP_FS = 13         # figure suptitle
LEG_FS = 9          # legends and in-panel annotation



# ---- canonical colours, so a model or a regime is the SAME colour in every figure ----
MODEL_COLOR = {
    'base0':       '#28658a',   # base                       (blue)
    'mt1k-0499':   '#c22f4f',   # Re=1000 matched fine-tune  (red)
    'mt2k-0599':   '#0f9e78',   # Re=2000 matched fine-tune  (green)
    'mt8k-0549':   '#d4770a',   # Re=8000 matched fine-tune  (orange, gate-failing)
    'r8kp02-0599': '#8a5cd6',   # Re=8000 repaired fine-tune (purple)
    'gt1k-0099':   '#b8399e',   # gated-dose fine-tune       (magenta)
}
# regimes as a cold -> hot ramp in the same two hues the model palette is built from
REGIME_COLOR = {1000: '#28658a', 1500: '#4a7fa5', 2000: '#7fb0cc', 3000: '#a8c4d4',
                4000: '#e0b0b8', 5000: '#e08a9c', 6000: '#d4667f', 7000: '#c8455f',
                8000: '#c22f4f'}
# linestyle encodes the STEERING, never the model
LS_UNGUIDED, LS_DIAL, LS_GATE = '-', '--', ':'


def apply(fontsize=BASE_FS, dpi=140):
    # fonttype 42 embeds TrueType subsets instead of Type 3 outlines: Type 3 text is rejected
    # by many publishers and rasterises badly when a figure is scaled in LaTeX.
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'cm', 'font.family': 'serif',
                         'font.size': fontsize, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.dpi': dpi, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                         'legend.frameon': False})


def shade_bands(ax):
    ax.axvspan(16, 32, color=SHADE_MID, zorder=0)
    ax.axvspan(32, 96, color=SHADE_HI, zorder=0)


def re_axis(ax, regs=(1000, 2000, 4000, 8000)):
    ax.set_xscale('log'); ax.set_xticks(list(regs))
    ax.set_xticklabels([(f'{r / 1000:g}k' if r % 1000 else f'{r // 1000}k') for r in regs])
    ax.minorticks_off()
