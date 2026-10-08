"""fig_frontier.py -- D1 noise dial, gamma >= 1: compact islands and an empty gap (reads results only)."""
import numpy as np
import nfp_plot as P

r, num = P.load()
x, g, m = r['x'], r['frontier_gamma'], r['frontier_m']
E, ne = r['frontier_edges'], r['frontier_n_edges']
fr = num['frontier_D1_gamma_ge_1']
g_wall, g_gap = fr['gamma_support_leaves_walls_interp'], fr['gamma_gap_opens_at_x0_interp']
L = x[-1]

fig, ax = P.new_fig()
P.density_map(fig, ax, x, g, m, 'crowd density m(x)', vmin=0.0, vmax=float(m.max()))
P.contour_lines(ax, x, g, m, [0.25, 0.5, 0.75], '%.2f')
# support edges: outer pair from the walls, inner pair from the gap opening at x = 0
k2 = ne >= 2
k4 = ne >= 4
outer_l = np.concatenate([[-L], E[k2, 0]])
outer_r = np.concatenate([[L], np.where(ne[k2] == 4, E[k2, 3], E[k2, 1])])
gy2 = np.concatenate([[g_wall], g[k2]])
inner_l = np.concatenate([[0.0], E[k4, 1]])
inner_r = np.concatenate([[0.0], E[k4, 2]])
gy4 = np.concatenate([[g_gap], g[k4]])
for xs, ys in ((outer_l, gy2), (outer_r, gy2), (inner_l, gy4), (inner_r, gy4)):
    ax.plot(xs, ys, color=P.CORAL, lw=1.8, zorder=5)
P.tag(ax, 1.47, 2.2, 'support edge', ha='left')
P.tag(ax, 0.0, 2.05, 'empty gap' + P.NL + '(m = 0)', color=P.NAVY)
P.tag(ax, 2.12, 2.3, 'empty', color=P.NAVY)
P.tag(ax, -2.12, 2.3, 'empty', color=P.NAVY)
P.tag(ax, 0.0, g_gap - 0.035, r'gap opens, $\gamma$ = %.3f' % g_gap, color=P.NAVY, fontsize=10, va='top')
P.linear_row_label(ax, g[0] + 0.004, -1.97)
ax.set_ylim(g[0], g[-1])
P.axis_labels(ax, r'noise exponent $\gamma$   (noise $\nu\, m^{\gamma-1}$)')
print(P.save(fig, __file__))
