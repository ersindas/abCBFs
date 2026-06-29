import numpy as np


class Estimator:
    def setupEstimator(self):

        # theta = [c_x, c_z, delta_g, delta_m, delta_J, delta_l]
        # delta_g, delta_m widened so the true mismatch (0.22, -0.32) sits inside
        self.theta_hat_low = np.array([-.1, -.1, -.25, -.35, -.01, -.005])
        self.theta_hat_high = np.array([.1, .1, .25, .35, .01, .005])
        self.n_theta = 6

        # start at the box midpoint -> least conservative rho(0)
        self.theta_hat = (self.theta_hat_low + self.theta_hat_high) / 2

        self.rho = np.maximum(
            np.abs(self.theta_hat - self.theta_hat_low),
            np.abs(self.theta_hat_high - self.theta_hat),
        )
        self.rho0 = self.rho.copy()

        # DREM filtered regression: Me_dot = -lambda_F Me + R^T R, Ye_dot = -lambda_F Ye + R^T y
        self.lambda_F = 1.0
        self.Me = np.zeros((self.n_theta, self.n_theta))
        self.Ye = np.zeros(self.n_theta)

        # causal normalization of chi = det(Me), running max of |chi|
        self.chi_scale = 1e-12
        self.chi_bar = 0.0

        self.gamma = 30.0 * np.ones(self.n_theta)

        self.theta_hat_dot = np.zeros(self.n_theta)
        self.rho_dot = np.zeros(self.n_theta)

    def regressionRows(self, x, u):
        # rows of y = R theta for the planar quadrotor
        th = x[2]
        vx = x[3]
        vz = x[4]
        F = u[0]
        M = u[1]

        R = np.array(
            [
                [-vx, 0.0, 0.0, F * np.sin(th), 0.0, 0.0],
                [0.0, -vz, -1.0, F * np.cos(th), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -M, F],
            ]
        )
        return R

    def regressionTargets(self, x, u, xdot_meas):
        # y = measured accel minus what the nominal model already explains
        th = x[2]
        F = u[0]
        M = u[1]
        y = np.array(
            [
                xdot_meas[3] - (np.sin(th) / self.m0) * F,
                xdot_meas[4] - (-self.g0 + (np.cos(th) / self.m0) * F),
                xdot_meas[5] - (-M / self.J0),
            ]
        )
        return y

    @staticmethod
    def adjugate(A):
        n = A.shape[0]
        Aadj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Aij = np.delete(np.delete(A, j, axis=0), i, axis=1)
                Aadj[i, j] = (-1) ** (i + j) * np.linalg.det(Aij)
        return Aadj

    def updateEstimator(self, x, u, xdot_meas, dt):
        # one euler step of DREM plus the certified bound rho_i
        R = self.regressionRows(x, u)
        y = self.regressionTargets(x, u, xdot_meas)

        self.Me = self.Me + dt * (-self.lambda_F * self.Me + R.T @ R)
        self.Ye = self.Ye + dt * (-self.lambda_F * self.Ye + R.T @ y)

        # mixing step
        chi = np.linalg.det(self.Me)
        YD = self.adjugate(self.Me) @ self.Ye

        # causal normalization keeps chi_bar bounded
        self.chi_scale = max(self.chi_scale, abs(chi))
        self.chi_bar = chi / self.chi_scale
        Y_bar = YD / self.chi_scale

        self.theta_hat_dot = self.gamma * self.chi_bar * (
            Y_bar - self.chi_bar * self.theta_hat
        )
        self.rho_dot = -self.gamma * self.chi_bar**2 * self.rho

        self.theta_hat = self.theta_hat + dt * self.theta_hat_dot
        self.rho = self.rho * np.exp(-self.gamma * self.chi_bar**2 * dt)

        # euler steps can drift slightly outside the box, clip back in
        self.theta_hat = np.clip(
            self.theta_hat, self.theta_hat_low, self.theta_hat_high
        )

        return self.theta_hat, self.rho
