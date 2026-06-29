import numpy as np
import cvxpy as cp


class Control:
    def setupControl(self):

        self.F_max = 20.  # [N]
        self.M_max = 20.  # [Nm]

        self.u_bounds = (
            (0, self.F_max),
            (-self.M_max, self.M_max),
        )

        self.thetad_max_p = np.deg2rad(80)

        # primary control gains
        wx = 2
        wz = 2.4
        zeta = 1.0
        wth = 12

        self.primaryParams = {
            "xg": 1.0,
            "zg": 1.0,
            "Kp_x": wx**2,
            "Kd_x": 2 * zeta * wx,
            "Kp_z": wz**2,
            "Kd_z": 2 * zeta * wz,
            "Kp_th": wth**2,
            "Kd_th": 2 * zeta * wth,
        }

        # critically damped, omega_n ~ 4 rad/s
        self.backupParams = {
            "th_r": np.deg2rad(25),
            "Kp_th": 4.0,
            "Kd_th": 2.0,
        }

    def primaryControl(self, x):
        x, z, th, xd, zd, thd = x

        xg = self.primaryParams["xg"]
        zg = self.primaryParams["zg"]

        Kp_x = self.primaryParams["Kp_x"]
        Kd_x = self.primaryParams["Kd_x"]

        Kp_z = self.primaryParams["Kp_z"]
        Kd_z = self.primaryParams["Kd_z"]

        Kp_th = self.primaryParams["Kp_th"]
        Kd_th = self.primaryParams["Kd_th"]

        ax = -Kp_x * (x - xg) - Kd_x * xd
        az = -Kp_z * (z - zg) - Kd_z * zd
        az = np.clip(az, -2.0 * self.g0, 2.0 * self.g0)

        th_des = np.arctan2(ax, self.g0 + az)
        th_des = np.clip(th_des, -self.thetad_max_p, self.thetad_max_p)

        F = self.m0 * np.sqrt(ax * ax + (self.g0 + az) * (self.g0 + az))
        F = np.clip(F, 0, self.F_max)

        # inner-loop PD on pitch
        M = -self.J0 * (-Kp_th * (th - th_des) - Kd_th * thd)
        M = np.clip(M, -self.M_max, self.M_max)

        u_des = np.array([F, M])
        return u_des

    def checkValidBackup(self):
        # vertex LMIs, anti-saturation, invariance, vertical-thrust, backup subset of safe set
        kth = self.backupParams["Kp_th"]
        kw = self.backupParams["Kd_th"]
        self.th_r = self.backupParams["th_r"]

        a_j_low = 1 / self.J0 + self.theta_hat_low[4]
        a_j_high = 1 / self.J0 + self.theta_hat_high[4]
        a_m_low = 1 / self.m0 + self.theta_hat_low[3]
        dg_high = self.theta_hat_high[2]
        dl_bar = max(abs(self.theta_hat_low[5]), abs(self.theta_hat_high[5]))

        def A_th(a):
            return np.array([[0.0, 1.0], [-a * kth, -a * kw]])

        # Q fixed at I -- letting the solver pick Q drives it to its lower bound
        n = 2
        P = cp.Variable((n, n), symmetric=True)
        t = cp.Variable()
        constraints = [
            P >> 1e-3 * np.eye(n),
            P << t * np.eye(n),
            A_th(a_j_low).T @ P + P @ A_th(a_j_low) << -np.eye(n),
            A_th(a_j_high).T @ P + P @ A_th(a_j_high) << -np.eye(n),
        ]
        prob = cp.Problem(cp.Minimize(t), constraints)
        prob.solve(solver=cp.SCS, eps=1e-9)

        self.P = P.value
        self.Q = np.eye(n)
        self.rho_th = 0.05

        if self.skip_backup_check:
            print("Skipped backup set and controller check. The backup controller is not verified!")
            return

        # Assumption 1: a_m, a_J must stay positive over the parameter box
        if not (a_m_low > 0 and a_j_low > 0):
            raise Exception("Parameter bounds allow a_m <= 0 or a_J <= 0.")

        for a in (a_j_low, a_j_high):
            lam = np.max(np.linalg.eigvalsh(A_th(a).T @ self.P + self.P @ A_th(a) + self.Q))
            if lam > 1e-7:
                raise Exception(f"Vertex LMI violated (residual {lam:.2e}).")

        cond1 = np.min(np.linalg.eigvalsh(self.P)) > 0.0
        if not cond1:
            raise Exception("P is not positive definite.")

        K_th = np.array([kth, kw])
        e1 = np.array([1, 0])
        b_th = np.array([0, 1])
        Pinv = np.linalg.inv(self.P)
        delta_th = np.sqrt(self.rho_th * e1 @ Pinv @ e1)
        delta_M = np.sqrt(self.rho_th * K_th @ Pinv @ K_th.T)
        self.delta_th = delta_th
        self.th_bar = self.th_r + delta_th

        cond2 = self.th_r - delta_th >= 0
        cond3 = self.th_r + delta_th <= np.sqrt(self.theta_max**2 - self.r_th)
        cond4 = delta_M <= self.M_max

        if not (cond2 and cond3 and cond4):
            raise Exception("Anti-saturation conditions not met.")

        lam_maxP = np.max(np.linalg.eigvalsh(self.P))
        lam_minQ = np.min(np.linalg.eigvalsh(self.Q))

        rho_min = lam_maxP * (
            (2 * np.linalg.norm(self.P @ b_th, ord=2) * self.F_max * dl_bar) / lam_minQ
        ) ** 2
        cond5 = self.rho_th >= rho_min
        if not cond5:
            raise Exception("Invariance condition (33) not met.")

        cond_vert = a_m_low * self.F_max * np.cos(self.th_bar) - self.g0 - dg_high >= 0
        if not cond_vert:
            raise Exception("Vertical thrust condition (34) not met.")

        cond6 = (
            np.exp(-self.kappa * self.r_x)
            + np.exp(-self.kappa * self.r_z)
            + np.exp(-self.kappa * self.r_th)
            <= 1
        )
        if not cond6:
            raise Exception("Backup set is not a subset of the safe set (35).")

        rho_max = min(
            self.th_r**2 / (e1 @ Pinv @ e1),
            (np.sqrt(self.theta_max**2 - self.r_th) - self.th_r) ** 2 / (e1 @ Pinv @ e1),
            self.M_max**2 / (K_th @ Pinv @ K_th.T),
        )
        if self.verbose:
            print(
                f"All conditions passed: Backup set and controller are valid.\n"
                f"  Delta_th = {np.degrees(delta_th):.2f} deg, "
                f"th_bar = {np.degrees(self.th_bar):.2f} deg, "
                f"Delta_M = {delta_M:.3f} Nm\n"
                f"  rho_th = {self.rho_th} in feasible interval "
                f"[{rho_min:.2e}, {rho_max:.4f}]"
            )
