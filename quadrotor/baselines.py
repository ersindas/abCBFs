"""Nominal bCBF and primary-controller comparison on the true model."""
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from model import Simulation
import safety
from experiment_config import configure_prior_and_truth

def run(mode, delta_m, out):
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
    configure_prior_and_truth(E)
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
    clock = time.perf_counter()
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
    indices = np.flatnonzero(arrays['used_backup'][:-1])
    summary = dict(
        mode=mode,
        completed=True,
        minimum_dense_h=float(hh.min()),
        minimum_dense_components=hc.min(axis=0).tolist(),
        backup_steps=int(len(indices)),
        elapsed_seconds=time.perf_counter() - clock,
    )
    (out / (mode + '_summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    safety.quadprog.solve_qp = original_solve
