import numpy as np
import time
import itertools
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

        # adaptive backup flow discretization (jitted RK4 rollout)
        self.dtau = 0.02
        self.n_steps_b = int(round(self.backupTime / self.dtau))   # 50 -> 51 pts
        # subsample interior constraint points for QP speed (terminal added too)
        self.con_stride = 3

        # flow-error-bound envelope
        self.flow_V = 12.0
        self.flow_Om = 6.0    # margin above the closed-loop |omega| (~4.2)
        self._flow_ready = False
        self._env_warned = False

    # ====================================================================== #
    #                          vanilla bCBF-QP                                #
    # ====================================================================== #
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

    # ====================================================================== #
    #                     flow-error-bound constants                         #
    # ====================================================================== #
    def _compute_flow_constants(self):
        # L_b, L_h, L_hb for the eps(tau,x,t) tightening, all certified in the
        # same weighted norm ||.||_T = diag(1,1,1,eps_v,eps_v,1), eps_v chosen
        # by a 1-D search that minimizes L_b
        g0, m0, J0 = self.g0, self.m0, self.J0
        kth = self.backupParams["Kp_th"]
        kw = self.backupParams["Kd_th"]
        th_max = self.theta_max
        Fmax = self.F_max
        V, Om, kap = self.flow_V, self.flow_Om, self.kappa

        lo = self.theta_hat_low
        hi = self.theta_hat_high
        corners = [np.array(cc) for cc in itertools.product(*zip(lo, hi))]
        ths = np.linspace(-th_max, th_max, 41)

        def Jcl(th, that):
            cx, cz, dg, dm, dJ, dl = that
            am, aJ = 1 / m0 + dm, 1 / J0 + dJ
            J = np.zeros((6, 6))
            J[0, 3] = J[1, 4] = J[2, 5] = 1.0
            J[3, 2] = am * np.cos(th) * Fmax
            J[3, 3] = -cx
            J[4, 2] = -am * np.sin(th) * Fmax
            J[4, 4] = -cz
            J[5, 2] = -aJ * kth
            J[5, 5] = -aJ * kw
            return J

        def mu(A):
            return np.max(np.linalg.eigvalsh(0.5 * (A + A.T)))

        def weighted_T(eps_v):
            return np.diag([1.0, 1.0, 1.0, eps_v, eps_v, 1.0])

        def Lb_for(Tmat, Tinv, th_grid):
            Lb_ = -1e9
            for th in th_grid:
                for that in corners:
                    Lb_ = max(Lb_, mu(Tmat @ Jcl(th, that) @ Tinv))
            return Lb_

        # 1-D search over the v_x,v_z weight minimizing L_b (coarse grid first)
        ths_coarse = ths[::4]
        best_Lb, best_eps_v = np.inf, 1.0
        for eps_v in np.linspace(0.02, 1.0, 30):
            T = weighted_T(eps_v)
            Tinv = np.linalg.inv(T)
            Lb_try = Lb_for(T, Tinv, ths_coarse)
            if Lb_try < best_Lb:
                best_Lb, best_eps_v = Lb_try, eps_v

        T = weighted_T(best_eps_v)
        Tinv = np.linalg.inv(T)
        Lb = Lb_for(T, Tinv, ths)        # refine at the chosen weight, full theta grid

        # L_h, L_hb = sup over envelope of ||grad h . T^-1||_2
        Lh = Lhb = 0.0
        for px in (self.px_min, self.px_min + 6):
            for pz in (self.pz_min, self.pz_min + 6):
                for th in ths[::4]:
                    for vx in (-V, 0, V):
                        for vz in (-V, 0, V):
                            for om in (-Om, 0, Om):
                                xx = np.array([px, pz, th, vx, vz, om])
                                Lh = max(Lh, np.linalg.norm(self.grad_hsoft(xx, kap) @ Tinv))
                                Lhb = max(Lhb, np.linalg.norm(self.grad_hsoft_b(xx, kap) @ Tinv))

        self.flow_T = T
        self.flow_Lb = float(Lb)
        self.flow_Lh = float(Lh)
        self.flow_Lhb = float(Lhb)
        self._flow_ready = True
        if self.verbose:
            print(f"[flow bound] eps_v={best_eps_v:.3f}  L_b={Lb:.3f}  L_h={Lh:.3f}  L_hb={Lhb:.3f}")

    def _envelope_check(self, x):
        if self._env_warned:
            return
        bad = (
            abs(x[2]) > self.theta_max + 1e-6
            or abs(x[3]) > self.flow_V
            or abs(x[4]) > self.flow_V
            or abs(x[5]) > self.flow_Om
        )
        if bad:
            self._env_warned = True
            if self.verbose:
                print(
                    f"[warn] state left the flow-bound envelope at step "
                    f"{self.curr_step}: theta={np.degrees(x[2]):.1f} deg, "
                    f"v=({x[3]:.2f},{x[4]:.2f}), omega={x[5]:.2f}. "
                    f"eps constants may no longer be valid."
                )

    # ====================================================================== #
    #                      robust adaptive bCBF-QP                            #
    # ====================================================================== #
    def adaptive_bCBF(self, x, u_des):
        # propagate the estimated flow, tighten each constraint by eps = L*delta_max,
        # then robustify against the current parametric uncertainty
        if not self._flow_ready:
            self._compute_flow_constants()
        self._envelope_check(x)

        th, vx, vz = x[2], x[3], x[4]
        sin_th, cos_th = np.sin(th), np.cos(th)

        theta_hat = np.asarray(self.theta_hat)
        rho = np.asarray(self.rho)
        theta_hat_dot = np.asarray(self.theta_hat_dot)
        rho_dot = np.asarray(self.rho_dot)

        n = self.n_steps_b
        dtau = self.dtau

        if not getattr(self, "_adapt_warmed", False):
            _ = self.adaptiveBackupFlow(
                x, theta_hat, rho, rho_dot, self.flow_Lb, self.flow_T, n, dtau
            )
            self._adapt_warmed = True
        tic = time.perf_counter()
        zs, Phis, Gams, deltas, Xis, Qs = self.adaptiveBackupFlow(
            x, theta_hat, rho, rho_dot, self.flow_Lb, self.flow_T, n, dtau
        )
        flowComputeTime = time.perf_counter() - tic

        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(zs)

        f_hat = self.f_x_hat(x, theta_hat)
        g_hat = self.g_x_hat(x, theta_hat)

        # box constraints: G u >= d
        G = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        d = np.array([0.0, -self.M_max, -self.F_max, -self.M_max])
        rows, rhs = [G], [d]

        # interior constraint points (subsampled) + terminal point
        idxs = list(range(1, n, self.con_stride))
        if idxs and idxs[-1] != n:
            idxs.append(n)
        elif not idxs:
            idxs = [n]

        for i in idxs:
            phi_i = zs[i]
            Phi_i = Phis[i]
            Gam_i = Gams[i]
            Xi_i = Xis[i]
            Q_i = Qs[i]
            terminal = i == n

            if terminal:
                hval = self.h_soft_b(phi_i, self.kappa)
                gradh = self.grad_hsoft_b(phi_i, self.kappa)
                L = self.flow_Lhb
                a_of = self.alpha_b
            else:
                hval = self.h_soft(phi_i, self.kappa)
                gradh = self.grad_hsoft(phi_i, self.kappa)
                L = self.flow_Lh
                a_of = self.alpha

            eps_i = L * deltas[i]
            deps_i = L * Q_i
            # d hbar(phi_i)/d x0 = grad h(phi_i) Phi_i - d(eps)/dx
            ell = gradh @ Phi_i - L * Xi_i            # (1x6)
            ell3, ell4, ell5 = ell[3], ell[4], ell[5]

            # constant part of d/dt h(phi_i) and constant robust terms
            a0 = (
                ell @ f_hat
                + gradh @ Gam_i @ theta_hat_dot      # moving-estimate channel
                - deps_i
                + a_of(hval - eps_i)
                - rho[0] * abs(ell3 * vx)
                - rho[1] * abs(ell4 * vz)
                - rho[2] * abs(ell4)
            )
            ag = ell @ g_hat                         # (2,)
            cu0 = rho[3] * abs(ell3 * sin_th + ell4 * cos_th) + rho[5] * abs(ell5)
            a1_0 = ag[0] - cu0                       # coeff of u0 (F)
            a1_1 = ag[1]                             # coeff of u1 (M)
            c_abs = rho[4] * abs(ell5)               # multiplies |u1|

            # a0 + a1_0 u0 + a1_1 u1 - c_abs |u1| >= 0  <=>  two linear cons:
            rows.append(np.array([[a1_0, a1_1 - c_abs],
                                  [a1_0, a1_1 + c_abs]]))
            rhs.append(np.array([-a0, -a0]))

        G_all = np.vstack(rows)
        d_all = np.concatenate(rhs)

        M = np.eye(2)
        q = u_des
        k_b = np.asarray(self.kb_vanilla(x, self.F_max, self.M_max, self.backupParams))
        try:
            tic = time.perf_counter()
            soltn = quadprog.solve_qp(M, q, G_all.T, d_all, 0)
            k_star = soltn[0]
            indicator_F = 1            # 1_F(x,t)=1: a safe u in U exists
            solverComputeTime = time.perf_counter() - tic
        except Exception:
            k_star = None
            indicator_F = 0            # 1_F(x,t)=0: fall back to the backup controller
            solverComputeTime = None
            if self.verbose and self.curr_step % 25 == 0:
                print(f"  adaptive QP infeasible at step {self.curr_step}; using backup u_b")

        # k_safe = 1_F k* + (1-1_F) k_b
        u_act = k_star if indicator_F else k_b

        intervening = np.linalg.norm(u_act - u_des) >= 0.0001
        return u_act, intervening, solverComputeTime, flowComputeTime
