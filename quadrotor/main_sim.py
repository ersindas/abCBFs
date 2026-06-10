"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import os
from pathlib import Path
import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics
from estimator import Estimator
from plotting_paper import PaperPlotter
import jax.numpy as jnp


class Simulation(ASIF, Control, Dynamics, Estimator):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        vanilla_flag=True,
        robust_flag=False,
        skip_backup_check = False,
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
        """
        Simulates trajectory for pre-specified number of timesteps and performs
        point-wise safety-critical control modifications if applicable.

        """
        x0 = self.x0
        total_steps = self.total_steps
        lenx = self.lenx

        # Tracking variables
        x_full = np.zeros((lenx, total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        solver_times, flow_times, avg_solver_t, max_solver_t, intervened = (
            [],
            [],
            [],
            [],
            [],
        )
        x_full[:, 0] = x0
        x_curr = x0

        # TODO: Initialize the parameter estimates

        # Main loop
        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            # Generate desired control
            u_des = self.primaryControl(x_curr[:lenx])
            u_des_full[:, i] = u_des

            # If safety check on, monitor control using active set invariance filter
            if self.safety_flag:
                if self.vanilla_flag:
                    u, boolean, sdt, fdt = self.vanilla_bCBF(x_curr, u_des)
                    solver_times.append(sdt)
                    flow_times.append(fdt)
                if self.robust_flag:
                    # TODO Make this function
                    # Should be similar to vanilla_bCBF but props. the flow differently, and uses the estimate
                    # Should take the current estimate as an input
                    u, boolean, sdt, fdt = self.adaptive_bCBF(x_curr, u_des)
                    solver_times.append(sdt)
                    flow_times.append(fdt)
                if boolean:
                    intervened.append(i)
            else:
                u = u_des

            u_act_full[:, i] = u

            # Propagate states 
            # TODO Modify this to integrate the parameter estimates with this since they are continuous
            x_curr = self.integrateState(
                x_curr,
                u,
                t,
                self.del_t,
                self.int_options,
            )
            x_full[:, i] = x_curr[:lenx]

            self.curr_step += 1

        if self.verbose and self.safety_flag:
            solver_times = [i for i in solver_times if i is not None]
            if solver_times:
                avg_solver_t = 1000 * np.average(solver_times)  # in ms
                max_solver_t = 1000 * np.max(solver_times)  # in ms
                print(f"Average solver time: {avg_solver_t:0.4f} ms")
                print(f"Maximum single solver time: {max_solver_t:0.4f} ms")

            avg_flow_t = 1000 * np.average(flow_times)  # in ms
            max_flow_t = 1000 * np.max(flow_times)  # in ms
            print(f"Average flow time: {avg_flow_t:0.4f} ms")
            print(f"Maximum single flow time: {max_flow_t:0.4f} ms")

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

    show_plots = True
    save_plots = False

    if save_plots:
        base_dir = os.path.dirname(__file__)
        folder = Path(base_dir, "plots")
        folder.mkdir(parents=True, exist_ok=True)

    safety_flag = True
    vanilla_flag = True
    robust_flag = False
    skip_backup_check = True

    env = Simulation(
        safety_flag=safety_flag,
        verbose=True,
        vanilla_flag=vanilla_flag,
        robust_flag=robust_flag,
        skip_backup_check=skip_backup_check,
    )
    print(
        "Running simulation with parameters:",
        "Safety:",
        env.safety_flag,
        "| Vanilla:",
        env.vanilla_flag,
        "| Robust:",
        env.robust_flag,
    )
    (
        x_full,
        total_steps,
        u_des_full,
        u_act_full,
        intervened,
        avg_solver_t,
        max_solver_t,
    ) = env.sim()

    PaperPlotter.plotter(
        x_full,
        u_act_full,
        intervened,
        u_des_full,
        env,
        save_plots=save_plots,
        show_plots=show_plots,
        u_plots=True,
    )
