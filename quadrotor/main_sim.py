from pathlib import Path
import argparse
import numpy as np
from scipy.integrate import solve_ivp

import safety
from safety import ASIF
from control import Control
from dynamics import Dynamics
from estimator import Estimator, estimator_rates
from predictor import StructuredBackupPredictor
from dual_qp import error_polytope, solve_lifted_qp


class Simulation(ASIF, Control, Dynamics, Estimator):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        vanilla_flag=True,
        robust_flag=False,
        skip_backup_check=False,
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


def regression_coefficients(x):
    phi0 = np.zeros((6, 6))
    phi0[3, 0], phi0[4, 1], phi0[4, 2] = (-x[3], -x[4], -1.0)
    phiu = np.zeros((2, 6, 6))
    phiu[0, 3, 3], phiu[0, 4, 3] = (np.sin(x[2]), np.cos(x[2]))
    phiu[0, 5, 5], phiu[1, 5, 4] = (1.0, -1.0)
    return phi0, phiu


def get_input(env, predictor, x, reserve=0.0):
    hbar, Jx, Jtheta, Jrho = predictor.values_and_jacobians(x, env.theta_hat, env.rho)
    hbar, Jx, Jtheta, Jrho = map(np.asarray, (hbar, Jx, Jtheta, Jrho))
    fhat, ghat = (env.f_x_hat(x, env.theta_hat), env.g_x_hat(x, env.theta_hat))
    phi0, phiu = regression_coefficients(x)
    target = hbar - reserve
    alpha = getattr(env, 'running_alpha_gain', 2.0) * target + target ** 3
    alpha[-1] = 10 * target[-1] + target[-1] ** 3
    a0 = Jx @ fhat + Jtheta @ env.theta_hat_dot + Jrho @ env.rho_dot + alpha
    au = Jx @ ghat
    c0 = Jx @ phi0
    cu = np.stack([Jx @ phiu[i] for i in range(2)], axis=2)
    A, b, _, _ = error_polytope(env.theta_hat, env.rho, env.theta_hat_low, env.theta_hat_high)
    result = solve_lifted_qp(
        a0[1:], au[1:], c0[1:], cu[1:], A, b,
        np.array([0.0, -env.M_max]), np.array([env.F_max, env.M_max]),
        env.primaryControl(x),
    )
    result['hbar'] = hbar
    result['qp_prediction_indices'] = np.arange(1, len(hbar))
    return result


def run_adaptive(
    output, dt=0.01, duration=4.0, prediction_dt=0.01,
    reserve=0.005, alpha_gain=2.0, delta_m=-0.37,
):
    output.mkdir(parents=True, exist_ok=True)
    env = Simulation(
        safety_flag=True, vanilla_flag=False, robust_flag=True,
        verbose=False, skip_backup_check=False,
    )
    env.running_alpha_gain = alpha_gain
    if delta_m != -0.37:
        raise ValueError('This declared experiment is delta_m=-.37 only.')
    x0 = np.array([7.0, 3.0, np.deg2rad(5.0), -2.0, -0.8, 0.1])
    rho_floor = 1e-08
    z = np.r_[x0, env.Me.ravel(), env.Ye, env.theta_hat, env.rho]
    pred = StructuredBackupPredictor(
        env.P, dtau=prediction_dt, n_steps=int(round(1 / prediction_dt)),
        theta_low=env.theta_hat_low, theta_high=env.theta_hat_high,
    )
    steps = int(round(duration / dt))
    records = {k: [] for k in [
        't', 'x', 'joint', 'u', 'ud', 'theta_hat', 'rho', 'hbar',
        'qp_success', 'qp_status', 'used_backup', 'qp_residual',
        'predicted_states', 'predicted_radii', 'moment_bound',
        'certificate_group', 'certificate_row_upper_bound',
    ]}
    dense_joint = []
    dense_t = []

    def assign(z):
        x = z[:6]
        Me = z[6:42].reshape(6, 6)
        Ye = z[42:48]
        env.theta_hat = z[48:54].copy()
        raw_rho = z[54:60].copy()
        env.theta_hat_dot, raw_rate = estimator_rates(Me, Ye, env.theta_hat, raw_rho, env.gamma)
        env.rho = np.sqrt(raw_rho ** 2 + rho_floor ** 2)
        env.rho_dot = raw_rho / env.rho * raw_rate
        return x, Me, Ye, raw_rate

    for k in range(steps + 1):
        t = k * dt
        x, _, _, _ = assign(z)
        q = get_input(env, pred, x, reserve)
        pp = pred.evaluate(x, env.theta_hat, env.rho)
        moment_bound = float(pp['all_time_moment_bound'])
        if moment_bound >= env.M_max:
            raise RuntimeError(f'Unsaturation certificate fails at t={t}')
        use_qp = q['success'] and q['hbar'].min() >= 0.0
        if not q['success'] and q['classification'] != 'primal_infeasible':
            raise RuntimeError(f"Unresolved solver failure: {q['status']}")
        if k == 0:
            if q['hbar'].min() <= reserve:
                raise RuntimeError('Initial tightened-set check failed')
            if not use_qp:
                raise RuntimeError(
                    'The declared initial condition requires a feasible QP; '
                    'initial backup is not forced.'
                )
        u = np.asarray(q['u'] if use_qp else env.kb_vanilla(x, env.F_max, env.M_max, env.backupParams))
        row = dict(
            t=t, x=x.copy(), joint=z.copy(), u=u, ud=env.primaryControl(x),
            theta_hat=env.theta_hat.copy(), rho=env.rho.copy(), hbar=q['hbar'],
            qp_success=q['success'], qp_status=q['status'], used_backup=not use_qp,
            qp_residual=q.get('min_robust_barrier_residual', np.nan),
            predicted_states=pp['states'], predicted_radii=pp['radii'],
            moment_bound=moment_bound,
            certificate_group=q.get('certificate_group', -1),
            certificate_row_upper_bound=q.get('certificate_row_upper_bound', np.nan),
        )
        for key, value in row.items():
            records[key].append(np.asarray(value))
        if k % 50 == 0 or k == steps:
            print(
                f"t={t:.2f}, backup={not use_qp}, Hmin={q['hbar'].min():.7g}, "
                f"h={env.h_soft(x, env.kappa):.7g}", flush=True,
            )
        if k == steps:
            break

        def rhs(ti, zi):
            xx, Me, Ye, raw_rate = assign(zi)
            xdot = env.propState(ti, xx, u)
            R = env.regressionRows(xx, u)
            y = env.regressionTargets(xx, u, xdot)
            return np.r_[
                xdot, (-env.lambda_F * Me + R.T @ R).ravel(),
                -env.lambda_F * Ye + R.T @ y, env.theta_hat_dot, raw_rate,
            ]

        sol = solve_ivp(rhs, (t, t + dt), z, rtol=1e-09, atol=1e-11, max_step=dt / 2, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        times = np.linspace(t, t + dt, 11)[:-1]
        dense_t.extend(times)
        dense_joint.extend(sol.sol(times).T)
        z = sol.y[:, -1]
    dense_t.append(duration)
    dense_joint.append(z.copy())
    arrays = {key: np.asarray(value) for key, value in records.items()}
    dj = np.asarray(dense_joint)
    arrays.update(
        dense_t=np.asarray(dense_t), dense_joint=dj, P=env.P, theta_true=env.bTheta,
        theta_low=env.theta_hat_low, theta_high=env.theta_hat_high,
        dt=dt, prediction_dt=prediction_dt, reserve=reserve, rho_floor=rho_floor,
        running_alpha_gain=alpha_gain, qp_omits_tau_zero=True, kappa=env.kappa,
        qp_prediction_indices=q['qp_prediction_indices'],
    )
    np.savez_compressed(output / 'adaptive.npz', **arrays)
    print(f"Saved adaptive simulation to {output / 'adaptive.npz'}")


def run_baseline(mode, delta_m, out):
    out.mkdir(parents=True, exist_ok=True)
    original_solve = safety.quadprog.solve_qp
    last = {}

    def record_solve(G, a, C, b, meq=0, *args, **kwargs):
        last.clear()
        last.update(success=False, status='not_returned', residual=np.nan)
        try:
            answer = original_solve(G, a, C, b, meq, *args, **kwargs)
            last.update(success=True, status='solved', residual=float(np.min(C.T @ answer[0] - b)))
            return answer
        except Exception as exc:
            last.update(success=False, status=type(exc).__name__ + ': ' + str(exc))
            raise
    safety.quadprog.solve_qp = record_solve
    E = Simulation(
        safety_flag=mode == 'nominal',
        vanilla_flag=mode == 'nominal',
        robust_flag=False,
        verbose=False,
        skip_backup_check=False,
    )
    if delta_m != -0.37:
        raise ValueError('This declared experiment is delta_m=-.37 only.')
    assert E.m0 == 1.0
    x0 = np.array([7.0, 3.0, np.deg2rad(5.0), -2.0, -0.8, 0.1])
    x = x0.copy()
    dt = 0.01
    duration = 4.0
    steps = 400
    records = {key: [] for key in [
        't',
        'x',
        'u',
        'ud',
        'qp_success',
        'qp_status',
        'qp_residual',
        'used_backup',
        'h_soft',
        'h_components',
    ]}
    dense_times = []
    dense_states = []
    for k in range(steps + 1):
        E.curr_step = k
        t = k * dt
        ud = np.asarray(E.primaryControl(x))
        if mode == 'nominal':
            u, intervention, solver_time, flow_time = E.vanilla_bCBF(x, ud)
            u = np.asarray(u)
        else:
            u = ud.copy()
            last.update(success=False, status='PrimaryController', residual=np.nan)
        row = dict(
            t=t,
            x=x.copy(),
            u=u,
            ud=ud,
            qp_success=last['success'],
            qp_status=last['status'],
            qp_residual=last['residual'],
            used_backup=mode == 'nominal' and (not last['success']),
            h_soft=E.h_soft(x, E.kappa),
            h_components=[E.h_x(x), E.h_z(x), E.h_theta(x)],
        )
        for key, value in row.items():
            records[key].append(value)
        if k % 100 == 0:
            print(mode, delta_m, 't', t, 'h', row['h_soft'], 'backup', row['used_backup'], flush=True)
        if k == steps:
            break
        sol = solve_ivp(
            lambda ti, xi: E.propState(ti, xi, u),
            (t, t + dt),
            x,
            rtol=1e-09,
            atol=1e-11,
            max_step=dt / 2,
            dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        tt = np.linspace(t, t + dt, 11)[:-1]
        dense_times.extend(tt)
        dense_states.extend(sol.sol(tt).T)
        x = sol.y[:, -1]
    dense_times.append(duration)
    dense_states.append(x.copy())
    dx = np.asarray(dense_states)
    tt = np.asarray(dense_times)
    hh = np.asarray([E.h_soft(xx, E.kappa) for xx in dx])
    hc = np.asarray([[E.h_x(xx), E.h_z(xx), E.h_theta(xx)] for xx in dx])
    arrays = {key: np.asarray(v) for key, v in records.items()}
    arrays.update(
        dense_t=tt,
        dense_x=dx,
        dense_h=hh,
        dense_h_components=hc,
        initial_x=x0,
        dt=dt,
        prediction_dt=E.del_t_b,
        backup_horizon=E.backupTime,
        P=E.P,
        theta_true=E.bTheta,
        theta_low=E.theta_hat_low,
        theta_high=E.theta_hat_high,
        nominal_theta=np.zeros(6),
        true_mass=1 / (1 / E.m0 + E.delta_m),
        delta_m=E.delta_m,
        kappa=E.kappa,
    )
    np.savez_compressed(out / (mode + '.npz'), **arrays)
    safety.quadprog.solve_qp = original_solve
    print(f"Saved {mode} simulation to {out / (mode + '.npz')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['adaptive', 'vanilla', 'nominal'], default='adaptive',
                        help='adaptive bCBF, nominal bCBF, or primary controller only')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('data'))
    args = parser.parse_args()

    if args.mode == 'adaptive':
        run_adaptive(args.output)
    else:
        mode = 'nominal' if args.mode == 'vanilla' else 'primary'
        run_baseline(mode, -.37, args.output)


if __name__ == '__main__':
    main()
