"""fig_congestion.py -- D3 congestion dial: q-exponential islands, effective edges (reads results only)."""
import numpy as np
import nfp_plot as P

r, num = P.load()
x, c, m = r['x'], r['congestion_c'], r['congestion_m']
E, ne = r['congestion_edges'], r['congestion_n_edges']
cg = num['congestion_D3']
c_gap = cg['c_gap_opens_interp_m0_eq_1e-6max']

fig, ax = P.new_fig()
P.density_map(fig, ax, x, c, m, 'crowd density m(x)', vmin=0.0, vmax=float(m.max()))
P.contour_lines(ax, x, c, m, [0.25, 0.5, 0.75], '%.2f')
k4 = ne >= 4
ax.plot(E[:, 0], c, color=P.CORAL, lw=1.8, zorder=5)
ax.plot(np.where(ne == 4, E[:, 3], E[:, 1]), c, color=P.CORAL, lw=1.8, zorder=5)
ax.plot(np.concatenate([[0.0], E[k4, 1]]), np.concatenate([[c_gap], c[k4]]), color=P.CORAL, lw=1.8, zorder=5)
ax.plot(np.concatenate([[0.0], E[k4, 2]]), np.concatenate([[c_gap], c[k4]]), color=P.CORAL, lw=1.8, zorder=5)
P.tag(ax, 1.62, 0.62, 'effective edge' + P.NL + r'$m = 10^{-6}\,\max m$', ha='left')
P.tag(ax, 0.0, 0.8, 'near-empty gap', color=P.NAVY)
P.tag(ax, 0.0, c_gap - 0.02, r'gap opens, $c$ = %.3f' % c_gap, color=P.NAVY, fontsize=10, va='top')
P.linear_row_label(ax, c[0] + 0.008, -2.45)
ax.set_ylim(c[0], c[-1])
P.axis_labels(ax, r'congestion exponent $c$   (cost $(\varepsilon+m)^{c}\, v^2/2$, $\varepsilon = 10^{-3}$)')
print(P.save(fig, __file__))
