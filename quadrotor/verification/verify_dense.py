"""Check physical safety and estimator bounds at every stored dense state.

Negative tightened membership is reported but is not itself a failure: backup
switching may preserve the true safe set outside the tightened set. These finite
checks do not prove safety between stored states.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = SCRIPT_DIR.parent/'data'
sys.path.insert(0, str(SCRIPT_DIR.parent))
from predictor import StructuredBackupPredictor
import jax
from verification_checks import (TOLERANCES, check, require_finite,
    parameter_checks, input_checks, save_summary, run_guarded)


def audit(path, out):
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(path)
    require_finite(d, ['dense_joint', 'dense_t', 'P', 'rho_floor', 'theta_true',
                      'theta_low', 'theta_high', 'u', 'dt'])
    z, t, P = d['dense_joint'], d['dense_t'], d['P']
    if z.ndim != 2 or z.shape != (len(t), 60) or len(t) == 0:
        raise ValueError('Dense joint history must have shape (number of times, 60)')
    if np.any(np.diff(t) <= 0) or float(d['dt']) <= 0 or float(d['rho_floor']) < 0:
        raise ValueError('Time grid, control step or bound floor is invalid')
    raw = z[:, 54:60]
    rho = np.sqrt(raw**2+float(d['rho_floor'])**2)
    pred = StructuredBackupPredictor(P, dtau=.005, n_steps=200,
        theta_low=d['theta_low'], theta_high=d['theta_high'])
    vbars = jax.jit(jax.vmap(pred.barriers))
    blocks = []
    for start in range(0, len(t), 200):
        zz = z[start:start+200]
        blocks.append(np.asarray(vbars(zz[:, :6], zz[:, 48:54], rho[start:start+200])))
    h = np.concatenate(blocks, axis=0)
    arg = np.unravel_index(np.argmin(h), h.shape)
    y = z[:, [2, 5]]-np.array([np.deg2rad(25), 0])
    V = np.einsum('bi,ij,bj->b', y, P, y)
    moment = pred.certificate['moment_gain']*np.maximum(
        np.sqrt(np.maximum(V, 0)), pred.certificate['forcing_radius'])
    # Independent evaluation of the actual smooth safe set (not Hbar).
    safe_components = np.column_stack([z[:, 0]-1, z[:, 1]-1,
                                      np.deg2rad(60)**2-z[:, 2]**2])
    actual_h = -logsumexp(-10*safe_components, axis=1)/10
    checks = {}
    contain = parameter_checks(checks, z[:, 48:54], rho, d['theta_true'],
                               d['theta_low'], d['theta_high'])
    input_checks(checks, d['u'])
    safe_tol = TOLERANCES['safety_absolute']
    check(checks, 'dense_physical_safety', actual_h.min() >= -safe_tol,
          minimum=float(actual_h.min()), tolerance=safe_tol)
    moment_limit = 20.-TOLERANCES['moment_strict_margin']
    check(checks, 'all_time_moment_certificate', moment.max() < moment_limit,
          maximum=float(moment.max()), strict_limit=moment_limit)
    check(checks, 'nonnegative_raw_error_radii', raw.min() >= -TOLERANCES['parameter_absolute'],
          minimum=float(raw.min()), tolerance=TOLERANCES['parameter_absolute'])
    check(checks, 'finite_audit_results',
          all(np.all(np.isfinite(value)) for value in (h, moment, actual_h, rho)))
    between = ~np.isclose(t/float(d['dt']), np.round(t/float(d['dt'])), atol=1e-7, rtol=0)
    bi = np.flatnonzero(between)
    ba = np.unravel_index(np.argmin(h[between]), h[between].shape) if len(bi) else None
    summary = dict(
        input=str(path), dense_time_dt=float(np.max(np.diff(t))) if len(t)>1 else None,
        prediction_dt=.005, num_states=len(t), num_constraints=h.shape[1],
        minimum_hbar=float(h.min()), minimum_hbar_time=float(t[arg[0]]),
        minimum_constraint_index=int(arg[1]),
        minimum_terminal=float(h[:, -1].min()),
        minimum_terminal_time=float(t[int(np.argmin(h[:, -1]))]),
        minimum_running=float(h[:, :-1].min()),
        minimum_between_updates_hbar=float(h[between].min()) if ba else None,
        minimum_between_updates_hbar_time=float(t[bi[ba[0]]]) if ba else None,
        minimum_between_updates_constraint_index=int(ba[1]) if ba else None,
        minimum_physical_h=float(actual_h.min()),
        minimum_physical_h_time=float(t[int(np.argmin(actual_h))]),
        minimum_physical_components=safe_components.min(axis=0).tolist(),
        maximum_moment_bound=float(moment.max()), maximum_parameter_error_excess=float(contain.max()),
        negative_hbar_count=int(np.sum(h<0)), minimum_raw_rho=float(raw.min()),
        parameter_containment_tolerance=TOLERANCES['parameter_absolute'],
        parameter_containment_within_tolerance=checks['parameter_containment']['passed'],
        strictly_nonpositive_containment_excess=bool(contain.max()<=0),
        maximum_containment_excess_time=float(t[np.unravel_index(np.argmax(contain), contain.shape)[0]]),
        maximum_containment_excess_component=int(np.unravel_index(np.argmax(contain), contain.shape)[1]),
        note='All saved dense closed-loop states and 5ms prediction points checked numerically. The numerical tolerances are not a containment proof or universal intersample theorem.')
    np.savez_compressed(out/'dense_membership_values.npz', t=t, hbar=h,
                        moment_bound=moment, physical_h=actual_h)
    return save_summary(summary, checks, out/'dense_membership_verification.json')


def run(path, out):
    return run_guarded(audit, path, out, 'dense_membership_verification.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=CANDIDATE_DIR/'adaptive/continuous_candidate.npz')
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.input, args.output) else 1)
