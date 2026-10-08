"""fig_herding.py -- D4 McKean-Vlasov dial: the herded branch beyond theta_c (reads results only)."""
import numpy as np
import nfp_plot as P

r, num = P.load()
x, th, m = r['x'], r['herding_theta'], r['herding_m']
tc = float(r['herding_theta_c'])
s, ts = r['herding_branch_s'], r['herding_branch_theta']
o = np.argsort(ts)
xb_t = np.concatenate([[th[0], tc], ts[o]])
xb_x = np.concatenate([[0.0, 0.0], s[o]])
sel = xb_t <= th[-1]

fig, ax = P.new_fig()
P.density_map(fig, ax, x, th, m, 'crowd density m(x)', vmin=0.0, vmax=float(m.max()))
P.contour_lines(ax, x, th, m, [0.5, 1.0, 1.5], '%.1f')
ax.plot(xb_x[sel], xb_t[sel], color=P.CORAL, lw=2.0, zorder=5)
ax.axhline(tc, color=P.GREY, lw=1.3, ls=(0, (6, 3)), zorder=4)
P.tag(ax, 0.55, 0.0236, r'mean $\bar x(\theta)$', ha='left')
P.tag(ax, -2.42, tc, r'$\theta_c$ = %.5f' % tc, color=P.NAVY, ha='left', va='bottom', fontsize=11)
P.tag(ax, -1.0, 0.0245, 'left well empties', color=P.NAVY, fontsize=10.5)
P.linear_row_label(ax, th[0] + 0.00025, -2.45)
ax.set_ylim(th[0], th[-1])
P.axis_labels(ax, r'drift strength $\theta$   (drift $-\theta\,(x-\bar x)$)')
print(P.save(fig, __file__))
