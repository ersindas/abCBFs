import numpy as np

class Estimator:

    def setupEstimator(self):
        self.theta_hat_low = np.array([-0.1, -0.1, -0.25, -0.37, -0.01, -0.005])
        self.theta_hat_high = np.array([0.1, 0.1, 0.25, 0.35, 0.01, 0.005])
        self.n_theta = 6
        self.theta_hat = (self.theta_hat_low + self.theta_hat_high) / 2
        self.rho = np.maximum(
            np.abs(self.theta_hat - self.theta_hat_low),
            np.abs(self.theta_hat_high - self.theta_hat),
        )
        self.rho0 = self.rho.copy()
        self.lambda_F = 1.0
        self.Me = np.zeros((self.n_theta, self.n_theta))
        self.Ye = np.zeros(self.n_theta)
        self.gamma = 30.0 * np.ones(self.n_theta)
        self.theta_hat_dot = np.zeros(self.n_theta)
        self.rho_dot = np.zeros(self.n_theta)

    def regressionRows(self, x, u):
        th = x[2]
        vx = x[3]
        vz = x[4]
        F = u[0]
        M = u[1]
        R = np.array([
            [-vx, 0.0, 0.0, F * np.sin(th), 0.0, 0.0],
            [0.0, -vz, -1.0, F * np.cos(th), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -M, F],
        ])
        return R

    def regressionTargets(self, x, u, xdot_meas):
        th = x[2]
        F = u[0]
        M = u[1]
        y = np.array([
            xdot_meas[3] - np.sin(th) / self.m0 * F,
            xdot_meas[4] - (-self.g0 + np.cos(th) / self.m0 * F),
            xdot_meas[5] - -M / self.J0,
        ])
        return y
MIXING_RANK_FREEZE = 1e-10
MIXING_RANK_FULL = 1e-08
MIXING_DETERMINANT_SCALE = 1e-12

def estimator_rates(Me, Ye, that, rho, gamma):
    """Normalized DREM rates with a smooth numerical rank gate.

    The same nonnegative gate multiplies both rates so a rank-deficient
    regression does not contract the error bound.
    """
    matrix = (np.asarray(Me) + np.asarray(Me).T) / 2
    matrix_scale = np.hypot(np.linalg.norm(matrix, 'fro'), 1e-12)
    scaled_matrix = matrix / matrix_scale
    eigenvalues = np.linalg.eigvalsh(scaled_matrix)
    if eigenvalues[0] <= MIXING_RANK_FREEZE:
        return (np.zeros_like(that), np.zeros_like(rho))
    transition = np.clip(
        (eigenvalues - MIXING_RANK_FREEZE) / (MIXING_RANK_FULL - MIXING_RANK_FREEZE),
        0.0,
        1.0,
    )
    activation = np.prod(transition ** 3 * (10.0 + transition * (-15.0 + 6.0 * transition)))
    log_chi = len(eigenvalues) * np.log(matrix_scale) + np.log(eigenvalues).sum()
    log_ratio = log_chi - np.log(MIXING_DETERMINANT_SCALE)
    normalized_gain = np.exp(-np.logaddexp(0.0, -2.0 * log_ratio))
    gain = np.asarray(gamma) * activation * normalized_gain
    residual = (np.asarray(Ye) - matrix @ that) / matrix_scale
    correction = np.linalg.solve(scaled_matrix, residual)
    return (gain * correction, -gain * rho)
