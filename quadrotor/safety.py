import time
import numpy as np
import quadprog
import jax.numpy as jnp
from scipy.integrate import solve_ivp


class Constraint:
    def alpha(self, x):
        return 2 * x + x**3

    def alpha_b(self, x):
        return 10 * x + x**3

    # ----------------------------- safe set ---------------------------------
    def h_x(self, x):
        return x[0] - self.px_min

    def grad_hx(self, x):
        return np.array([1, 0, 0, 0, 0, 0])

    def h_z(self, x):
        return x[1] - self.pz_min

    def grad_hz(self, x):
        return np.array([0, 1, 0, 0, 0, 0])

    def h_theta(self, x):
        return self.theta_max**2 - x[2] ** 2

    def grad_htheta(self, x):
        return np.array([0, 0, -2 * x[2], 0, 0, 0])

    @staticmethod
    def _softmin(vals, grads, kappa):
        # numerically stable soft-min and its gradient:
        # h = m - log(sum_i exp(-kappa (h_i - m))) / kappa, m = min_i h_i
        # grad = sum_i w_i grad_i, w_i = softmin weights
        vals = np.asarray(vals, dtype=float)
        m = vals.min()
        w = np.exp(-kappa * (vals - m))
        sw = w.sum()
        h = m - np.log(sw) / kappa
        g = np.zeros_like(grads[0], dtype=float)
        for wi, gi in zip(w, grads):
            g = g + wi * gi
        return h, g / sw

    def h_soft(self, x, kappa):
        return self._softmin(
            [self.h_x(x), self.h_z(x), self.h_theta(x)],
            [self.grad_hx(x), self.grad_hz(x), self.grad_htheta(x)],
            kappa,
        )[0]

    def grad_hsoft(self, x, kappa):
        return self._softmin(
            [self.h_x(x), self.h_z(x), self.h_theta(x)],
            [self.grad_hx(x), self.grad_hz(x), self.grad_htheta(x)],
            kappa,
        )[1]

    # ---------------------------- backup set --------------------------------
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
        # un-normalized pitch ellipsoid, keeps grad ~ O(1)
        y = np.array([x[2] - self.th_r, x[5]])
        return self.rho_th - y.T @ self.P @ y

    def grad_htheta_b(self, x):
        y = np.array([x[2] - self.th_r, x[5]])
        dhdy = -2.0 * y.T @ self.P
        return np.array([0, 0, dhdy[0], 0, 0, dhdy[1]])

    def _hb_lists(self, x):
        vals = [self.h_x_b(x), self.h_z_b(x), self.h_theta_b(x),
                self.h_vx_b(x), self.h_vz_b(x)]
        grads = [self.grad_hx_b(x), self.grad_hz_b(x), self.grad_htheta_b(x),
                 self.grad_hvx_b(x), self.grad_hvz_b(x)]
        return vals, grads

    def h_soft_b(self, x, kappa):
        vals, grads = self._hb_lists(x)
        return self._softmin(vals, grads, kappa)[0]

    def grad_hsoft_b(self, x, kappa):
        vals, grads = self._hb_lists(x)
        return self._softmin(vals, grads, kappa)[1]


class ASIF(Constraint):
    def setupASIF(self):

        # backup properties (shared by vanilla and adaptive for a fair test)
        self.backupTime = 1.0
        self.del_t_b = 2 * self.del_t
        self.tspan_b = np.arange(0, self.backupTime + self.del_t_b, self.del_t_b)
        self.backupTrajs = []
        self.backupTrajs_d = []
        self.backup_save_N = 5
        self.mu_d = 0.0

        # safety constraint
        self.kappa = 10.0
        self.px_min = 1.0    # wall, the active constraint in this demo
        self.pz_min = 1.0    # floor
        self.theta_max = np.deg2rad(60)

        # backup constraints
        self.r_x = 1.0
        self.r_z = 1.0
        self.r_th = 0.30                       # rad^2 pitch margin
        self.th_r = self.backupParams["th_r"]

    def vanilla_bCBF(self, x, u_des):
        # bCBF-QP, nominal model, no parameter adaptation
        M = np.eye(2)
        q = u_des

        G = np.vstack((np.eye(2), -np.eye(2)))
        h = np.zeros(4)
        h[1] = -self.M_max
        h[2] = -self.F_max
        h[3] = -self.M_max

        rtapoints = len(self.tspan_b)
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)
        new_x = np.concatenate((x, S[:, :, 0].flatten()))
        t_step = (0.0, self.tspan_b[-1])

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

        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x_nominal(x)
        gx_0 = self.g_x_nominal(x)

        for i in range(1, rtapoints):
            h_phi = self.h_soft(phi[i, :], self.kappa) - self.mu_d
            gradh_phi = self.grad_hsoft(phi[i, :], self.kappa)
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0
            h_temp_i = -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi))

            if i == 1:
                g_temp = g_temp_i
                h_temp = h_temp_i
            else:
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

            if i == rtapoints - 1:
                hb_phi = self.h_soft_b(phi[i, :], self.kappa)
                gradhb_phi = self.grad_hsoft_b(phi[i, :], self.kappa)
                h_temp_i = -((gradhb_phi @ S[:, :, i]) @ fx_0 + self.alpha_b(hb_phi))
                g_temp_i = (gradhb_phi.T @ S[:, :, i]) @ gx_0
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

        G = np.vstack([G, g_temp])
        h = np.vstack([h.reshape((-1, 1)), h_temp])

        d = h.reshape((len(h),))
        k_b = np.asarray(self.kb_vanilla(x, self.F_max, self.M_max, self.backupParams))
        try:
            tic = time.perf_counter()
            soltn = quadprog.solve_qp(M, q, G.T, d, 0)
            k_star = soltn[0]
            indicator_F = 1            # 1_F(x,t)=1: a safe u in U exists
            toc = time.perf_counter()
            solverComputeTime = toc - tic
        except Exception:
            k_star = None
            indicator_F = 0            # 1_F(x,t)=0: fall back to the backup controller
            solverComputeTime = None
            if self.verbose:
                print("no soltn (vanilla)")

        # k_safe = 1_F k* + (1-1_F) k_b
        u_act = k_star if indicator_F else k_b

        intervening = np.linalg.norm(u_act - u_des) >= 0.0001
        return u_act, intervening, solverComputeTime, flowComputeTime
