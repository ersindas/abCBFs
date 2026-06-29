import os
from pathlib import Path
import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics
from estimator import Estimator


class Simulation(ASIF, Control, Dynamics, Estimator):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        vanilla_flag=True,
        robust_flag=False,
        skip_backup_check=False,
    ) -> None:

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.vanilla_flag = vanilla_flag
        self.robust_flag = robust_flag
        self.skip_backup_check = skip_backup_check

        self.setupControl()
        self.setupDynamics()
        self.setupASIF()
        self.setupEstimator()
        self.checkValidBackup()

    def sim(self):
        x0 = self.x0
        total_steps = self.total_steps
        lenx = self.lenx

        x_full = np.zeros((lenx, total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        solver_times, flow_times, avg_solver_t, max_solver_t, intervened = (
            [], [], [], [], [],
        )
        x_full[:, 0] = x0
        x_curr = x0

        # only meaningful when robust_flag is True
        self.theta_hat_hist = np.zeros((self.n_theta, total_steps))
        self.rho_hist = np.zeros((self.n_theta, total_steps))
        self.maxratio_hist = np.zeros(total_steps)
        self.theta_hat_hist[:, 0] = self.theta_hat
        self.rho_hist[:, 0] = self.rho
        self.maxratio_hist[0] = np.max(np.abs(self.theta_hat - self.bTheta) / self.rho)

        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            u_des = self.primaryControl(x_curr[:lenx])
            u_des_full[:, i] = u_des

            if self.safety_flag:
                if self.robust_flag:
                    u, boolean, sdt, fdt = self.adaptive_bCBF(x_curr, u_des)
                elif self.vanilla_flag:
                    u, boolean, sdt, fdt = self.vanilla_bCBF(x_curr, u_des)
                else:
                    u, boolean, sdt, fdt = u_des, False, None, 0.0
                solver_times.append(sdt)
                flow_times.append(fdt)
                if boolean:
                    intervened.append(i)
            else:
                u = u_des

            u_act_full[:, i] = u

            # online DREM estimator update (ideal xdot measurement)
            if self.robust_flag:
                xdot_meas = self.propState(t, x_curr, u)
                self.updateEstimator(x_curr, u, xdot_meas, self.del_t)
            self.theta_hat_hist[:, i] = self.theta_hat
            self.rho_hist[:, i] = self.rho
            self.maxratio_hist[i] = np.max(
                np.abs(self.theta_hat - self.bTheta) / np.maximum(self.rho, 1e-12)
            )

            x_curr = self.integrateState(x_curr, u, t, self.del_t, self.int_options)
            x_full[:, i] = x_curr[:lenx]
            self.curr_step += 1

        if self.verbose and self.safety_flag:
            st = [s for s in solver_times if s is not None]
            if st:
                avg_solver_t = 1000 * np.average(st)
                max_solver_t = 1000 * np.max(st)
                print(f"Average solver time: {avg_solver_t:0.4f} ms")
                print(f"Maximum single solver time: {max_solver_t:0.4f} ms")
            ft = [f for f in flow_times if f is not None]
            if ft:
                print(f"Average flow time: {1000*np.average(ft):0.4f} ms")
                print(f"Maximum single flow time: {1000*np.max(ft):0.4f} ms")
            if self.robust_flag:
                print(f"Max |theta_hat - theta|/rho over run: "
                      f"{np.max(self.maxratio_hist):0.3f}  (certified if <= 1)")

        return (
            x_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "adaptive", "nominal"],
                        default="adaptive")
    args = parser.parse_args()

    if args.mode == "vanilla":
        safety_flag, vanilla_flag, robust_flag, skip = True, True, False, False
    elif args.mode == "adaptive":
        safety_flag, vanilla_flag, robust_flag, skip = True, False, True, False
    else:  # nominal (no safety filter)
        safety_flag, vanilla_flag, robust_flag, skip = False, False, False, True

    env = Simulation(
        safety_flag=safety_flag,
        verbose=True,
        vanilla_flag=vanilla_flag,
        robust_flag=robust_flag,
        skip_backup_check=skip,
    )
    print(
        "Running:", args.mode,
        "| Safety:", env.safety_flag,
        "| Vanilla:", env.vanilla_flag,
        "| Robust:", env.robust_flag,
    )
    (
        x_full, total_steps, u_des_full, u_act_full,
        intervened, avg_solver_t, max_solver_t,
    ) = env.sim()

    h_traj = np.array([env.h_soft(x_full[:, k], env.kappa) for k in range(total_steps)])
    # h_soft underestimates min(h_x,h_z,h_theta) near constraint corners
    # (softmin <= min always), so judge SAFE/VIOLATED against the hard
    # constraints, not the soft surrogate used inside the QP
    h_hard_traj = np.array([
        min(env.h_x(x_full[:, k]), env.h_z(x_full[:, k]), env.h_theta(x_full[:, k]))
        for k in range(total_steps)
    ])
    print(f"min_t h_soft(x(t)) = {h_traj.min():.4f}  (smooth QP surrogate, "
          f"underestimates true safety margin near constraint corners)")
    print(f"min_t min(h_x,h_z,h_theta)(x(t)) = {h_hard_traj.min():.4f}  "
          f"(true hard constraints; safe iff >= 0; "
          f"{'SAFE' if h_hard_traj.min() >= -1e-6 else 'VIOLATED'})")
