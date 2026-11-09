"""Parameter bounds and true mass used in the quadrotor example."""
import numpy as np
TRUE_DELTA_M = -0.37
LOWER_DELTA_M = -0.37
UPPER_DELTA_M = 0.35

def configure_prior_and_truth(env):
    old_low = env.theta_hat_low.copy()
    old_high = env.theta_hat_high.copy()
    old_true = env.bTheta.copy()
    env.theta_hat_low = old_low.copy()
    env.theta_hat_high = old_high.copy()
    env.theta_hat_low[3] = LOWER_DELTA_M
    assert env.theta_hat_high[3] == UPPER_DELTA_M
    env.theta_hat = (env.theta_hat_low + env.theta_hat_high) / 2
    env.rho = np.maximum(
        np.abs(env.theta_hat - env.theta_hat_low),
        np.abs(env.theta_hat_high - env.theta_hat),
    )
    env.rho0 = env.rho.copy()
    env.theta_hat_dot = np.zeros_like(env.theta_hat)
    env.rho_dot = np.zeros_like(env.rho)
    env.delta_m = TRUE_DELTA_M
    env.bTheta = old_true.copy()
    env.bTheta[3] = env.delta_m
    assert np.array_equal(np.delete(env.theta_hat_low, 3), np.delete(old_low, 3))
    assert np.array_equal(env.theta_hat_high, old_high)
    assert np.array_equal(np.delete(env.bTheta, 3), np.delete(old_true, 3))
    assert np.all(env.theta_hat_low <= env.bTheta) and np.all(env.bTheta <= env.theta_hat_high)
    assert np.all(np.abs(env.bTheta - env.theta_hat) <= env.rho + 1e-15)
    assert np.all(env.rho <= env.theta_hat_high - env.theta_hat_low)
    assert np.max(np.abs(env.Me)) == 0 and np.max(np.abs(env.Ye)) == 0
    env.checkValidBackup()
    return env
