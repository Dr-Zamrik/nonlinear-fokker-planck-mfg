"""fig_tails.py -- D1 noise dial, gamma <= 1: heavy tails, log10 m (reads results only)."""
import numpy as np
import nfp_plot as P

r, num = P.load()
x, g, lm = r['x'], r['tails_gamma'], r['tails_log10m']
p = r['tails_exponent']
x0, x1 = (float(v) for v in r['tails_fit_window'])
tl = num['tails_D1_gamma_le_1']
vmin = -8.0
Z = np.clip(lm, vmin, None)

fig, ax = P.new_fig()
P.density_map(fig, ax, x, g, Z, r'$\log_{10}\, m(x)$', vmin=vmin, vmax=float(Z.max()))
P.contour_lines(ax, x, g, Z, [-6, -4, -2, -1], '%d', colors=[P.GREY])
ax.set_ylim(g[0], g[-1])          # gamma = 1 (linear game) at the bottom, heavier tails upward
for xs in (x0, x1, -x0, -x1):
    ax.axvline(xs, color=P.CORAL, lw=1.3, ls=(0, (5, 3)), zorder=5)
P.tag(ax, 0.5 * (x0 + x1), 0.285, 'tail fit window' + P.NL + r'$m \propto |x|^{-p}$', fontsize=10.5)
for gv in (0.97, 0.8, 0.6, 0.4):
    i = int(np.argmin(np.abs(g - gv)))
    P.tag(ax, 0.5 * (x0 + x1), g[i], r'$p$ = %.1f' % p[i], color=P.NAVY, fontsize=10.5)
P.linear_row_label(ax, g[0] - 0.008, -2.45)
P.tag(ax, 0.0, g[-1] + 0.008, r'top row $\gamma$ = %.3f; the branch folds at $\gamma \approx$ %.3f'
      % (g[-1], tl['lower_end']['last_converged_gamma_natural_continuation']['1600']), color=P.NAVY, fontsize=10, va='top')
P.axis_labels(ax, r'noise exponent $\gamma$   (noise $\nu\, m^{\gamma-1}$)')
print(P.save(fig, __file__))
