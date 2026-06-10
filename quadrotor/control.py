"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import numpy as np
import cvxpy as cp

class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.F_max = 20.  # [N]
        self.M_max = 20. # [Nm]

        self.u_bounds = (
            (0, self.F_max),
            (-self.M_max, self.M_max),
        )

        self.thetad_max_p = np.deg2rad(80)

        ############ PRIMARY ############
        # Primary control gains
        wx = 2
        wz = 2.4
        zeta = 1.0
        wth = 12

        self.primaryParams = {
            "xg": 0.0,
            "zg": 5,
            "Kp_x": wx**2,
            "Kd_x": 2 * zeta * wx,
            "Kp_z": wz**2,
            "Kd_z": 2 * zeta * wz,
            "Kp_th": wth**2,
            "Kd_th": 2 * zeta * wth,
        }

        ############ BACKUP ############
        # Backup control gains
        zeta = 1.0*1
        wth = 12.0*1
        
        self.backupParams = {
            "th_r": np.deg2rad(25),
            "Kp_th": wth**2,
            "Kd_th": 2 * zeta * wth,
        }

    def primaryControl(self, x):
        """
        Primary controller producing desired control at each step.

        """
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

        # Limit acceleration
        az = np.clip(az, -2.0 * self.g0, 2.0 * self.g0)

        # Desired accelerations to desired pitch and thrust
        th_des = np.arctan2(ax, self.g0 + az)
        th_des = np.clip(th_des, -self.thetad_max_p, self.thetad_max_p)

        F = self.m0 * np.sqrt(ax * ax + (self.g0 + az) * (self.g0 + az))
        F = np.clip(F, 0, self.F_max)

        # Inner-loop PD for the pitch dynamics
        M = -self.J0 * (-Kp_th * (th - th_des) - Kd_th * thd)
        M = np.clip(M, -self.M_max, self.M_max)

        u_des = np.array([F, M])
        return u_des
    
    def checkValidBackup(self):
        """
        Check that the backup controller is valid.

        """

        a_j_low = 1/self.J0 + self.theta_hat_low[4]
        a_j_high =  1/self.J0 + self.theta_hat_high[4]
        A_th_low = np.array(
            [[0., 1.], [-a_j_low*self.backupParams["Kp_th"], -a_j_low*self.backupParams["Kd_th"]]]
        )
        A_th_high = np.array(
            [[0., 1.], [-a_j_high*self.backupParams["Kp_th"], -a_j_high*self.backupParams["Kd_th"]]]
        )

        n = A_th_low.shape[0]
        P = cp.Variable((n, n), symmetric=True)
        Q = cp.Variable((n, n), symmetric=True)

        constraints = [
            P >> 1e-4 * np.eye(n),           
            Q >> 1e-4 * np.eye(n),           
            A_th_low.T @ P + P @ A_th_low + Q << 0,      
            A_th_high.T @ P + P @ A_th_high + Q << 0,     
        ]

        prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
        prob.solve(solver=cp.SCS)

        self.rho_th = .005

        self.P = 10*P.value
        self.Q = 10*Q.value

        if self.skip_backup_check:
            print("Skipped backup set and controller check. The backup controller is not verified!")
        else:
            cond1 = np.min(np.linalg.eigvalsh(self.P)) > 0.0
            if not cond1:
                raise Exception("P is not positive definite.")

            K_th = np.array([self.backupParams["Kp_th"], self.backupParams["Kd_th"]])
            e1 = np.array([1,0])
            b_th = np.array([0,1])
            delta_th = np.sqrt(self.rho_th*e1.T@np.linalg.inv(self.P)@e1)
            delta_M = np.sqrt(self.rho_th*K_th@np.linalg.inv(self.P)@K_th.T)

            cond2 = self.th_r - delta_th >= 0
            cond3 = self.th_r + delta_th <= np.sqrt(self.theta_max**2 - self.r_th)
            cond4 = delta_M <= self.M_max

            if not (cond2 and cond3 and cond4):
                raise Exception("Anti-saturation conditions not met.")
            
            lam_maxP = np.max(np.linalg.eigvalsh(self.P))
            lam_minQ = np.min(np.linalg.eigvalsh(self.Q))

            delta_l_min = self.theta_hat_low[5]
            delta_l_max = self.theta_hat_high[5]

            cond5 = self.rho_th >= lam_maxP*((2*np.linalg.norm(self.P@b_th, ord=2)*self.F_max*max(abs(delta_l_min),abs(delta_l_max)))/(lam_minQ))**2
            
            if not (cond5):
                raise Exception("Invariance condition not met.")

            cond6 = np.exp(-self.kappa*self.r_x) + np.exp(-self.kappa*self.r_z) +  np.exp(-self.kappa*self.r_th) <= 1

            if not (cond6):
                raise Exception("Backup set is not a subset of the safe set.")

            print("All conditions passed: Backup set and controller are valid.")