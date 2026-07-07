import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12.7,
    "legend.handlelength": 1.2,
    "axes.linewidth": 0.75,
    "lines.linewidth": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,            # embed truetype, editable text in the pdf
})
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = os.path.dirname(__file__)
d = np.load(os.path.join(HERE, "fig_data.npz"))
dt = float(d["dt"]); wall = float(d["px_min"]); floor = float(d["pz_min"])
Fmax = float(d["F_max"]); Mmax = float(d["M_max"])
xg = float(d["xg"]); zg = float(d["zg"])
xa, uaa = d["xa"], d["uaa"]
xv = d["xv"]
xp = d["xp"]                               # primary controller, true model
th, bT = d["th"], d["bTheta"]
snap_est, snap_true = d["snap_est"], d["snap_true"]
t_a = np.arange(xa.shape[1]) * dt

ka = xa.shape[1]                           # adaptive: full run, settles near goal
# vanilla: truncate just past the first crash (wall or floor), not the
# global min -- past the crash the flight is uncontrolled and can drift
# back to a lower x later and drag in the whole tail
_viol = np.where((xv[0] < wall) | (xv[1] < floor))[0]
kv = (int(_viol[0]) if len(_viol) else int(np.argmin(xv[0]))) + 6
xa_t, xv_t = xa[:, :ka], xv[:, :kv]
nf = 28                                    # flow-snapshot horizon shown (~0.55s)

C_BCBF, C_ADA, C_EST = "#0e7490", "#6c3a8e", "#1f9d55"
C_PRI = "#34495e"
C_CONSTRAINT = "#555555"
C_GOAL = "#b9770e"
C_DRONE = "#c2185b"   # drone-snapshot icons, one dominant color
UNSAFE, SAFE = "#fbe6e2", "#e3f2fd"

fig = plt.figure(figsize=(8.5, 7.0), constrained_layout=True)
fig.set_constrained_layout_pads(hspace=0.06, wspace=0.04)
gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 0.42, 0.85])


def draw_quad(ax, x, z, theta, L=0.22, col=C_DRONE):
    a = np.array([np.cos(theta), -np.sin(theta)])
    tdir = np.array([np.sin(theta), np.cos(theta)])
    p = np.array([x, z]); p1, p2 = p - L * a, p + L * a
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=2.0,
            solid_capstyle="round", zorder=6)
    for pe in (p1, p2):
        ax.plot([pe[0], pe[0] + 0.5 * L * tdir[0]],
                [pe[1], pe[1] + 0.5 * L * tdir[1]], color=col, lw=1.4, zorder=6)
        ax.add_patch(plt.Circle(pe + 0.5 * L * tdir, 0.055, color=col, zorder=6))


# ----------------------- top: x-z plane, wall on the left -------------------
ax = fig.add_subplot(gs[0, :])
xlo, xhi, zlo, zhi = 0.5, 7.4, 0.5, 3.5
ax.set_facecolor(UNSAFE)
ax.add_patch(Rectangle((wall, floor), xhi - wall, zhi - floor,
                        facecolor=SAFE, edgecolor="none", zorder=0.5))
ax.vlines(wall, floor, zhi, color=C_CONSTRAINT, lw=1.6, zorder=2)
ax.hlines(floor, wall, xhi, color=C_CONSTRAINT, lw=1.6, zorder=2)
TXT_Z = 10   # keep all top-panel text above the trajectories
up_05mm = mtransforms.ScaledTranslation(0, 1.0 / 25.4, fig.dpi_scale_trans)
ax.text(wall + 0.06, zhi - 0.06, r"$p_x^{{\rm min}}$", color=C_CONSTRAINT,
        ha="left", va="top", fontsize=17, zorder=TXT_Z,
        transform=ax.transData + up_05mm)
ax.text(3.0, floor + 0.07, r"$p_z^{{\rm min}}$", color=C_CONSTRAINT,
        ha="left", va="bottom", fontsize=17, zorder=TXT_Z)
ax.text(3.5, 0.75, "unsafe", color=C_CONSTRAINT, fontsize=15, ha="center", zorder=TXT_Z)
ax.text(6.4, 2.44, "safe set", color=C_CONSTRAINT, fontsize=15, ha="center", zorder=TXT_Z)
ax.text(6.4, 2.18, r"$\mathcal{C}_{\rm S}$", color=C_CONSTRAINT, fontsize=15, ha="center", zorder=TXT_Z)

ax.plot(xg, zg, marker="*", ms=14, color=C_GOAL,
        markeredgecolor="k", markeredgewidth=0.5, zorder=TXT_Z)
ax.annotate("goal", (xg, zg), xytext=(0, -16), textcoords="offset points",
            fontsize=14, ha="center", color=C_GOAL, zorder=TXT_Z)

# backup-flow snapshots, drawn above the main trajectory lines (zorder=5) so
# they stay visible where the recovery loop crosses them
for j, (ze, zt) in enumerate(zip(snap_est, snap_true)):
    le = r"$\widehat{\varphi}_{\rm b}(\tau,\mathbf{x},t)$" if j == 0 else None
    lt = r"$\varphi_{\rm b}(\tau,\mathbf{x})$" if j == 0 else None
    ax.plot(ze[:nf, 0], ze[:nf, 1], color=C_EST, lw=1.3, zorder=6.5, label=le)
    ax.plot(zt[:nf, 0], zt[:nf, 1], color="k", lw=1.0, ls=(0, (4, 2)), zorder=6.5, label=lt)

ax.plot(xp[0], xp[1], color=C_PRI, lw=1.8, ls=(0, (5, 2)), zorder=3.6,
        label="primary controller")
ax.plot(xv_t[0], xv_t[1], color=C_BCBF, lw=2.3, zorder=4, label="nominal bCBF")
ax.plot(xv_t[0, -1], xv_t[1, -1], "x", color=C_BCBF, ms=9, mew=2.4, zorder=7)
ax.plot(xa_t[0], xa_t[1], color=C_ADA, lw=2.3, zorder=5, label="robust adaptive bCBF")
for tt in [0.0, 0.30, 0.60, 0.90, 1.15, 1.35]:
    k = min(int(tt / dt), ka - 1)
    draw_quad(ax, xa[0, k], xa[1, k], xa[2, k])
draw_quad(ax, xa[0, -1], xa[1, -1], xa[2, -1])
ax.plot(xa[0, 0], xa[1, 0], "o", color="k", ms=4, zorder=TXT_Z)
ax.annotate(r"$t=0$", (xa[0, 0], xa[1, 0]), xytext=(2, 9 + 1.0 * 72 / 25.4),
            textcoords="offset points", fontsize=14, ha="center", zorder=TXT_Z)
ax.plot(xa[0, -1], xa[1, -1], "o", color="k", ms=4, zorder=TXT_Z)
ax.annotate(rf"$t={t_a[-1]:.0f}$", (xa[0, -1], xa[1, -1]), xytext=(26, 4),
            textcoords="offset points", fontsize=14, ha="left", zorder=TXT_Z)

ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi)
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.set_xlabel(r"position, $p_x$ (m)"); ax.set_ylabel(r"position, $p_z$ (m)")
ax.legend(loc="lower right", framealpha=0.95, handlelength=1.6)

# ----------------------- row 2: control inputs -------------------------------
ax = fig.add_subplot(gs[1, :])
for lev in (Fmax, -Mmax, Mmax, 0):
    ax.axhline(lev, color="0.55", lw=0.8, ls=":")
ax.plot(t_a, uaa[0], color="#d35400", lw=1.7, label=r"$F$")
ax.plot(t_a, uaa[1], color="#2471a3", lw=1.7, label=r"$M$")
ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"$u$")
ax.set_ylim(-Mmax - 3, Fmax + 3)
ax.set_xlim(t_a[0], t_a[-1])
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.legend(loc="lower right", framealpha=0.95, ncol=2, handlelength=1.6)

# ----------------------- row 3, col 1: large parameter estimates ------------
labels = [r"$c_x$", r"$c_z$", r"$\delta_g$", r"$\delta_m$",
          r"$\delta_J$", r"$\delta_\ell$"]
cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
tmax = 1.2
mask = t_a <= tmax

ax = fig.add_subplot(gs[2, 0])
for i in (0, 2, 3):                        # c_x, delta_g, delta_m
    ax.plot(t_a[mask], th[i][mask], color=cols[i], lw=1.8, label=labels[i], zorder=3)
    ax.axhline(bT[i], color=cols[i], lw=1.0, ls=":", zorder=2)
ax.set_xlabel(r"time (s)"); ax.set_ylabel(r"$\hat{\theta}_i$")
ax.set_xlim(0, tmax)
ax.set_ylim(-0.36, 0.26)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.legend(loc="center", bbox_to_anchor=(0.75, -0.12), bbox_transform=ax.transData,
          ncol=3, frameon=False)

# ----------------------- row 3, col 2: small parameter estimates, zoomed ----
# c_z sits right under c_x (both 0.08); delta_J, delta_l are smaller still --
# all three are illegible on the left panel's scale, hence the zoom
ax = fig.add_subplot(gs[2, 1])
for i in (1, 4, 5):                        # c_z, delta_J, delta_l
    ax.plot(t_a[mask], th[i][mask], color=cols[i], lw=1.8, label=labels[i], zorder=3)
    ax.axhline(bT[i], color=cols[i], lw=1.0, ls=":", zorder=2)
ax.set_xlabel(r"time (s)")                 # y-label omitted, same quantity as left panel
ax.set_xlim(0, tmax)
ax.set_ylim(-0.008, 0.092)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.04))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.legend(loc="center", bbox_to_anchor=(0.75, 0.045), bbox_transform=ax.transData,
          ncol=3, frameon=False)

fig.savefig(os.path.join(HERE, "demo_fig.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "demo_fig.png"), dpi=200,
            bbox_inches="tight")
print("saved demo_fig.pdf and demo_fig.png")
