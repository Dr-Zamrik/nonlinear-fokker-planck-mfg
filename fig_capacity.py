"""fig_capacity.py -- D2 capacity dial: Fermi-Dirac plateaus (reads results only)."""
import numpy as np
import nfp_plot as P

r, num = P.load()
x, y, m = r['x'], r['capacity_y'], r['capacity_m']
E, wid = r['capacity_plateau_edges'], r['capacity_plateau_width']
cap = num['capacity_D2']

fig, ax = P.new_fig()
P.density_map(fig, ax, x, y, m, 'crowd density m(x)', vmin=0.0, vmax=float(m.max()))
P.contour_lines(ax, x, y, m, [0.25, 0.5, 0.75], '%.2f')
k = wid > 0
y0 = cap['y_plateau_appears_interp_maxm_eq_0p95K']
for j in range(4):
    seed = 0.5 * (E[k, 0][0] + E[k, 1][0]) if j < 2 else 0.5 * (E[k, 2][0] + E[k, 3][0])
    ax.plot(np.concatenate([[seed], E[k, j]]), np.concatenate([[y0], y[k]]), color=P.CORAL, lw=1.8, zorder=5)
P.tag(ax, 1.0, 2.62, 'plateau' + P.NL + r'$m \geq 0.95\,K$', color=P.NAVY, fontsize=11)
P.tag(ax, 1.36, 2.3, 'plateau edge', ha='left')
P.tag(ax, -1.0, 2.62, 'plateau', color=P.NAVY, fontsize=11)
P.linear_row_label(ax, y[0] + 0.02, -2.45)
ax.set_ylim(y[0], y[-1])
P.axis_labels(ax, r'capacity load $y$ = linear peak / $K$')
print(P.save(fig, __file__))
