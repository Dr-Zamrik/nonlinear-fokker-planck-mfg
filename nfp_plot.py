"""nfp_plot.py -- house style for the five contour figures (reads nothing, solves nothing)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, TEAL, GREY, CORAL = '#1e3a5f', '#2a9d8f', '#8a94a6', '#c0504d'
CMAP = LinearSegmentedColormap.from_list('nfp_seq', ['#ffffff', TEAL, NAVY])
LABEL, TICK = 12.5, 11
NL = chr(10)   # newline without a backslash in the scripts
HALO = [pe.withStroke(linewidth=3.2, foreground='white')]


def load():
    r = np.load(os.path.join(HERE, 'results.npz'))
    with open(os.path.join(HERE, 'numbers.json')) as f:
        num = json.load(f)
    return r, num


def new_fig():
    fig, ax = plt.subplots(figsize=(12.4, 6.4), dpi=170, facecolor='white')
    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK)
    return fig, ax


def density_map(fig, ax, x, dial, Z, cbar_label, vmin=None, vmax=None):
    """smooth continuous fill (Gouraud-shaded mesh) + colourbar."""
    X, Y = np.meshgrid(x, dial)
    pc = ax.pcolormesh(X, Y, Z, shading='gouraud', cmap=CMAP, vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label(cbar_label, fontsize=LABEL)
    cb.ax.tick_params(labelsize=TICK)
    cb.outline.set_visible(False)
    ax.set_xlim(x[0], x[-1])
    return pc, cb


def contour_lines(ax, x, dial, Z, levels, fmt, colors=None, label_color=NAVY):
    cs = ax.contour(x, dial, Z, levels=levels, colors=colors or [NAVY], linewidths=0.9, alpha=0.95,
                    linestyles='solid')
    lab = ax.clabel(cs, inline=True, fontsize=10, fmt=fmt)
    for t in lab:
        t.set_color(label_color)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground='white')])
    return cs


def axis_labels(ax, ylabel, xlabel='position x'):
    ax.set_xlabel(xlabel, fontsize=LABEL)
    ax.set_ylabel(ylabel, fontsize=LABEL)


def linear_row_label(ax, y0, x_left, above=True):
    ax.text(x_left, y0, 'linear game', fontsize=11, color=NAVY, ha='left', va='bottom' if above else 'top',
            path_effects=HALO, zorder=6)


def tag(ax, x, y, s, color=CORAL, **kw):
    kw.setdefault('fontsize', 11)
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    return ax.text(x, y, s, color=color, path_effects=HALO, zorder=7, **kw)


def save(fig, script_file):
    out = os.path.splitext(os.path.basename(script_file))[0] + '.png'   # the cwd: zz_paper runs each script in its figures folder
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out
