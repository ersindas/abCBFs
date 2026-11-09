"""Check the finite sufficient inequalities in the accompanying proof.

This checks constants, not sampled trajectories or sampled boundary points.
It uses the recorded backup matrix. Q=0.99999 I absorbs its rounding error.
"""
import json
import argparse
from pathlib import Path
import sys
import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))
from experiment_config import LOWER_DELTA_M, UPPER_DELTA_M

P = np.array([[1.313529508294, 0.031510380339],
              [0.031510380339, 0.066620482092]])
Q = 0.99999 * np.eye(2)
kappa, rho_theta = 10.0, 0.05
F_max, M_max = 20.0, 20.0
theta_ref = np.deg2rad(25.0)
K = np.array([4.0, 2.0])
a_J_bounds = (3.99, 4.01)
# Use the same declared mass prior as the active experiment, not the older
# -.35 lower endpoint used before the experiment was extended to -.37.
a_m_lower, gravity_upper = 1.0 + LOWER_DELTA_M, 10.06
drag_upper, thrust_moment_error = 0.1, 0.005

P_eigenvalues = np.linalg.eigvalsh(P)
Pinv = np.linalg.inv(P)
pitch_radius = np.sqrt(rho_theta * Pinv[0, 0])
pitch_lower, pitch_upper = theta_ref - pitch_radius, theta_ref + pitch_radius
moment_bound = np.sqrt(rho_theta * K @ Pinv @ K)
lmi_slack = []
for a_J in a_J_bounds:
    A = np.array([[0.0, 1.0], [-4.0 * a_J, -2.0 * a_J]])
    lmi_slack.append(float(np.linalg.eigvalsh(-A.T @ P - P @ A - Q).min()))
acceleration_lower = np.array([
    a_m_lower * F_max * np.sin(pitch_lower),
    a_m_lower * F_max * np.cos(pitch_upper) - gravity_upper,
])
lambda_Q = np.linalg.eigvalsh(Q).min()
beta = 2 * F_max * thrust_moment_error * np.linalg.norm(P[:, 1])
a = lambda_Q / (2 * P_eigenvalues[-1])
b = beta**2 / (2 * lambda_Q)
v_min = -np.log1p(-np.exp(-kappa * rho_theta)) / kappa
epsilon = drag_upper / kappa * np.exp(
    -1 - kappa * (acceleration_lower - v_min) / drag_upper)
Gprime_at_one = -v_min + a * (rho_theta + 1 / kappa) - b
boundary_derivative_lower = a * rho_theta - b - epsilon.sum()

checks = dict(
    ordered_mass_prior=bool(LOWER_DELTA_M <= UPPER_DELTA_M),
    positive_inverse_mass=bool(a_m_lower > 0),
    positive_definite_P=bool(P_eigenvalues.min() > 0),
    positive_vertex_LMI_slack=bool(min(lmi_slack) > 0),
    pitch_interval_admissible=bool(0 < pitch_lower < pitch_upper < np.pi/2),
    unsaturated_backup_moment=bool(moment_bound < M_max),
    positive_acceleration_slack=bool(min(acceleration_lower-v_min) > 0),
    decreasing_G=bool(Gprime_at_one < 0),
    positive_smooth_boundary_derivative=bool(boundary_derivative_lower > 0),
)
summary = dict(
    parameter_source='experiment_config.py',
    delta_m_prior=[float(LOWER_DELTA_M), float(UPPER_DELTA_M)],
    inverse_mass_lower=float(a_m_lower),
    Q_min_eigenvalue=float(lambda_Q),
    P_eigenvalues=P_eigenvalues.tolist(),
    vertex_LMI_minimum_slacks=lmi_slack,
    pitch_interval=[float(pitch_lower), float(pitch_upper)],
    moment_bound=float(moment_bound),
    acceleration_lower=acceleration_lower.tolist(),
    minimum_velocity_on_smooth_boundary=float(v_min),
    exponential_remainders=epsilon.tolist(),
    Gprime_at_one=float(Gprime_at_one),
    smooth_backup_boundary_derivative_lower=float(boundary_derivative_lower),
    certificate_passed=all(checks.values()),
    checks=checks,
    failed_checks=[name for name, passed in checks.items() if not passed],
    limitation='Finite floating-point sufficient inequalities, not interval arithmetic.',
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=Path(__file__).with_name('smooth_backup_certificate.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['certificate_passed'] else 1)
