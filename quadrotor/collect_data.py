import os
import numpy as np
from main_sim import Simulation


def backup_u(env, x):
    th, thd = x[2], x[5]
    kp, kd = env.backupParams["Kp_th"], env.backupParams["Kd_th"]
    M = np.clip(kp * (th - env.th_r) + kd * thd, -env.M_max, env.M_max)
    return np.array([env.F_max, M])


def true_backup_flow(env, x0, n, dtau):
    # rolls out the TRUE dynamics under the backup controller from x0
    xs = np.zeros((n + 1, env.lenx)); xs[0] = x0
    x = x0.copy()
    for k in range(n):
        def fdyn(xx):
            return env.propState(0.0, xx, backup_u(env, xx))
        k1 = fdyn(x)
        k2 = fdyn(x + 0.5 * dtau * k1)
        k3 = fdyn(x + 0.5 * dtau * k2)
        k4 = fdyn(x + dtau * k3)
        x = x + (dtau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[k + 1] = x
    return xs


# ---- adaptive run ----------------------------------------------------------
ada = Simulation(safety_flag=True, vanilla_flag=False, robust_flag=True,
                 verbose=False, skip_backup_check=False)
xa, nsa, uda, uaa, iva, *_ = ada.sim()
ha = np.array([ada.h_soft(xa[:, k], ada.kappa) for k in range(nsa)])

# estimated backup-flow snapshots saved every backup_save_N steps
est_trajs = [np.asarray(z) for z in ada.backupTrajs]
save_N = ada.backup_save_N
# pick a handful of snapshots spread over the descent (where backup is active)
save_steps = np.arange(len(est_trajs)) * save_N
want_t = [0.10, 0.30, 0.50, 0.70, 0.90]   # avoid t~0.05, too close to t=0 to read
sel = [int(np.argmin(np.abs(save_steps * ada.del_t - tw))) for tw in want_t]
sel = sorted(set(sel))
snap_est, snap_true, snap_t = [], [], []
for s in sel:
    ze = est_trajs[s]
    x0 = np.asarray(ze[0])
    zt = true_backup_flow(ada, x0, ada.n_steps_b, ada.dtau)
    snap_est.append(ze); snap_true.append(zt)
    snap_t.append(save_steps[s] * ada.del_t)

# ---- primary-only trajectory on the TRUE (uncertain) model, no filter -----
def primary_only_true_flow(env, x0, N, dtp):
    xp = np.zeros((env.lenx, N)); xp[:, 0] = x0.copy()
    for k in range(N - 1):
        x = xp[:, k]; u = env.primaryControl(x)
        def fdyn(xx):
            return env.propState(0.0, xx, u)
        k1 = fdyn(x); k2 = fdyn(x + 0.5 * dtp * k1)
        k3 = fdyn(x + 0.5 * dtp * k2); k4 = fdyn(x + dtp * k3)
        xp[:, k + 1] = x + dtp / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xp

xp = primary_only_true_flow(ada, ada.x0, xa.shape[1], ada.del_t)
print(f"primary on true model ends at ({xp[0,-1]:.2f},{xp[1,-1]:.2f}) "
      f"-> goal ({ada.primaryParams['xg']},{ada.primaryParams['zg']})")

# ---- vanilla run -----------------------------------------------------------
van = Simulation(safety_flag=True, vanilla_flag=True, robust_flag=False,
                 verbose=False, skip_backup_check=False)
xv, nsv, udv, uav, ivv, *_ = van.sim()
hv = np.array([van.h_soft(xv[:, k], van.kappa) for k in range(nsv)])

print("adaptive min h =", round(float(ha.min()), 4),
      "| vanilla min h =", round(float(hv.min()), 4))
print("max certified ratio =", round(float(ada.maxratio_hist.max()), 3))
print("theta_hat final vs true:")
names = ["c_x", "c_z", "d_g", "d_m", "d_J", "d_l"]
for i, nm in enumerate(names):
    print(f"  {nm}: hat={ada.theta_hat_hist[i,-1]:+.4f}  true={ada.bTheta[i]:+.4f}"
          f"  rho={ada.rho_hist[i,-1]:.4f}")

np.savez(os.path.join(os.path.dirname(__file__), "fig_data.npz"),
         xa=xa, ha=ha, uaa=uaa, uda=uda,
         th=ada.theta_hat_hist, rho=ada.rho_hist, mr=ada.maxratio_hist,
         bTheta=ada.bTheta, dt=ada.del_t,
         xv=xv, hv=hv, uav=uav, xp=xp,
         pz_min=ada.pz_min, px_min=ada.px_min, F_max=ada.F_max, M_max=ada.M_max,
         xg=ada.primaryParams["xg"], zg=ada.primaryParams["zg"],
         theta_max=ada.theta_max,
         snap_est=np.array(snap_est), snap_true=np.array(snap_true),
         snap_t=np.array(snap_t))
print("saved fig_data.npz with", len(snap_est), "backup-flow snapshots at t =",
      [round(s, 2) for s in snap_t])
