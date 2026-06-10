"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import numpy as np
import time
import quadprog
import jax.numpy as jnp
from scipy.integrate import solve_ivp


class Constraint:
    def alpha(self, x):
        return 2 * x + x**3

    def alpha_b(self, x):
        return 10 * x + x**3

    def h_x(self, x):
        return x[0] - self.px_min

    def grad_hx(self, x):
        return np.array([1, 0, 0, 0, 0, 0])

    def h_z(self, x):
        return x[1] - self.pz_min

    def grad_hz(self, x):
        return np.array([0, 1, 0, 0, 0, 0])

    def h_theta(self, x):
        return self.theta_max**2 - x[2]**2
    
    def grad_htheta(self,x):
        return np.array([0, 0, -2*x[2], 0, 0, 0])
    
    def h_soft(self,x,kappa):
        return -(np.log(np.exp(-kappa*self.h_x(x)) + np.exp(-kappa*self.h_theta(x)) + np.exp(-kappa*self.h_z(x))))/kappa
    
    def grad_hsoft(self,x,kappa):
        h1_val = self.h_x(x)
        h2_val = self.h_z(x)
        h3_val = self.h_theta(x)
        gradh1_val = self.grad_hx(x)
        gradh2_val = self.grad_hz(x)
        gradh3_val = self.grad_htheta(x)
        w_grad = np.exp(-kappa * h1_val) * gradh1_val + np.exp(-kappa * h2_val) * gradh2_val+ np.exp(-kappa * h3_val) * gradh3_val
        w_hb = np.exp(-kappa * h1_val)  + np.exp(-kappa * h2_val)  + np.exp(-kappa * h3_val)  
        return w_grad / w_hb
    
    def h_vx_b(self, x):
        return x[3]

    def grad_hvx_b(self, x):
        return np.array([0, 0, 0, 1, 0, 0])
    
    def h_vz_b(self, x):
        return x[4]

    def grad_hvz_b(self, x):
        return np.array([0, 0, 0, 0, 1, 0])
    
    def h_x_b(self, x):
        return x[0] - self.px_min - self.r_x

    def grad_hx_b(self, x):
        return np.array([1, 0, 0, 0, 0, 0])

    def h_z_b(self, x):
        return x[1] - self.pz_min - self.r_z

    def grad_hz_b(self, x):
        return np.array([0, 1, 0, 0, 0, 0])

    def h_theta_b(self, x):
        y = np.array([x[2] - self.th_r, x[5]])
        return self.rho_th - y.T @ self.P @ y
    
    def grad_htheta_b(self,x):
        y = np.array([x[2] - self.th_r, x[5]])
        dhdy = -2.0 * y.T @ self.P
        return  np.array([0, 0, dhdy[0], 0, 0, dhdy[1]])
    
    def h_soft_b(self,x,kappa):
        return -(np.log(np.exp(-kappa*self.h_x_b(x)) + np.exp(-kappa*self.h_theta_b(x)) + np.exp(-kappa*self.h_z_b(x)) + np.exp(-kappa*self.h_vx_b(x)) + np.exp(-kappa*self.h_vz_b(x))))/kappa
    
    def grad_hsoft_b(self,x,kappa):
        h1_val = self.h_x_b(x)
        h2_val = self.h_z_b(x)
        h3_val = self.h_theta_b(x)
        h4_val = self.h_vx_b(x)
        h5_val = self.h_vz_b(x)
        gradh1_val = self.grad_hx_b(x)
        gradh2_val = self.grad_hz_b(x)
        gradh3_val = self.grad_htheta_b(x)
        gradh4_val = self.grad_hvx_b(x)
        gradh5_val = self.grad_hvz_b(x)
        w_grad = np.exp(-kappa * h1_val) * gradh1_val + np.exp(-kappa * h2_val) * gradh2_val + np.exp(-kappa * h3_val) * gradh3_val + np.exp(-kappa * h4_val) * gradh4_val+ np.exp(-kappa * h5_val) * gradh5_val
        w_hb = np.exp(-kappa * h1_val)  + np.exp(-kappa * h2_val)  + np.exp(-kappa * h3_val)  + np.exp(-kappa * h4_val)   + np.exp(-kappa * h5_val)  
        return w_grad / w_hb
     

class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        # self.backupTime = 6.75  # [sec] (total backup time)
        self.backupTime = 1.0
        self.del_t_b = 2 * self.del_t
        self.tspan_b = np.arange(0, self.backupTime + self.del_t_b, self.del_t_b)
        self.backupTrajs = []
        self.backupTrajs_d = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.mu_d = 0.0

        # Safety constraint
        self.kappa = 10.0
        self.px_min = 1.0
        self.pz_min = 1.0
        self.theta_max = np.deg2rad(60)

        # Backup constraints
        self.r_x = 1.0
        self.r_z = 1.0
        self.r_th = np.deg2rad(45)
        self.th_r = self.backupParams["th_r"]

    def vanilla_bCBF(self, x, u_des):
        """
        bCBF-QP.

        """

        # QP objective function
        M = np.eye(2)
        q = u_des

        # QP actuation constraints
        G = np.vstack((np.eye(2), -np.eye(2)))
        h = np.zeros(4)
        h[1] = -self.M_max
        h[2] = -self.F_max
        h[3] = -self.M_max

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate system under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        t_step = (0.0, self.tspan_b[-1])

        #   Warm up JIT
        _ = self.fb_vanilla_jac_jax_new(x)
        tic = time.perf_counter()
        soltn = solve_ivp(
            self.propMainBackup_vanilla_jax,
            t_step,
            new_x,
            method="RK45",
            rtol=self.int_options["rtol"],
            atol=self.int_options["atol"],
            t_eval=self.tspan_b,
        )
        toc = time.perf_counter()
       
        backupFlow = soltn.y[:, :]

        flowComputeTime = toc - tic

        phi[:, :] = backupFlow[:lenx, :].T
        S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x_nominal(x)
        gx_0 = self.g_x_nominal(x)

        # Construct constraints
        for i in range(1, rtapoints):
            h_phi = self.h_soft(phi[i, :], self.kappa) - self.mu_d
            gradh_phi = self.grad_hsoft(
                phi[i, :], self.kappa
            )
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0
            h_temp_i = -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi))

            if i == 1:
                g_temp = g_temp_i
                h_temp = h_temp_i
            else:
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

            # Make sure last point is in the backup set
            if i == rtapoints - 1:
                hb_phi = self.h_soft_b(phi[i, :],self.kappa)
                gradhb_phi = self.grad_hsoft_b(phi[i, :],self.kappa)

                h_temp_i = -((gradhb_phi @ S[:, :, i]) @ fx_0 + self.alpha_b(hb_phi))
                g_temp_i = (gradhb_phi.T @ S[:, :, i]) @ gx_0

                # Append constraint
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

        # Append constraints
        G = np.vstack([G, g_temp])
        h = np.vstack([h.reshape((-1, 1)), h_temp])

        # Solve QP
        d = h.reshape((len(h),))
        try:
            tic = time.perf_counter()
            soltn = quadprog.solve_qp(M, q, G.T, d, 0)
            active_constraint = soltn[5]
            print(active_constraint)
            u_act = soltn[0]
            toc = time.perf_counter()
            solverComputeTime = toc - tic
        except:
            u_act = self.kb_vanilla(x, self.F_max, self.M_max, self.backupParams)
            solverComputeTime = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solverComputeTime, flowComputeTime
