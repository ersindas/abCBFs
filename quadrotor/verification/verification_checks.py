"""Numerical audit policy shared by the two saved-run verifiers.

These tolerances cover floating-point arithmetic and the independently checked
RK4 integration errors. They are not margins proving continuous-time safety.
The tightened set is allowed to become negative when the backup policy is used;
physical safety, parameter containment and true backup reachability are required.
"""
from pathlib import Path
import json
import numpy as np

TOLERANCES = {
    'safety_absolute': 1e-8,
    'parameter_absolute': 1e-10,
    'bound_absolute': 1e-7,
    'bound_relative': 1e-7,
    'input_absolute': 1e-7,
    'prediction_absolute': 2e-6,
    'saved_prediction_absolute': 1e-9,
    'moment_strict_margin': 1e-8,
}


def check(checks, name, passed, **details):
    checks[name] = {'passed': bool(passed), **details}


def require_finite(data, keys):
    for name in keys:
        value = np.asarray(data[name])
        if not np.all(np.isfinite(value)):
            raise ValueError(f'{name} contains a non-finite value')


def parameter_checks(checks, estimates, radii, truth, low, high):
    tol = TOLERANCES['parameter_absolute']
    excess = np.abs(estimates-truth)-radii
    check(checks, 'parameter_containment', np.max(excess) <= tol,
          maximum_excess=float(np.max(excess)), tolerance=tol)
    check(checks, 'nonnegative_error_radii', np.min(radii) >= 0,
          minimum_radius=float(np.min(radii)))
    check(checks, 'ordered_parameter_prior', np.all(low <= high))
    check(checks, 'truth_in_parameter_prior',
          np.all(truth >= low-tol) and np.all(truth <= high+tol))
    check(checks, 'estimates_in_parameter_prior',
          np.all(estimates >= low-tol) and np.all(estimates <= high+tol))
    return excess


def input_checks(checks, inputs):
    tol = TOLERANCES['input_absolute']
    lower, upper = np.array([0., -20.]), np.array([20., 20.])
    excess = np.maximum(lower-inputs, inputs-upper)
    check(checks, 'applied_input_bounds', np.max(excess) <= tol,
          maximum_excess=float(np.max(excess)), tolerance=tol)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save_summary(summary, checks, output_path):
    summary.update(
        verification_passed=all(v['passed'] for v in checks.values()),
        failed_checks=[name for name, value in checks.items() if not value['passed']],
        checks=checks,
        numerical_tolerances=TOLERANCES,
        tightened_membership_policy='Diagnostic only: temporary negative tightened membership is permitted with backup switching. Physical safety and true backup reachability remain required.',
    )
    result = _json_value(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result['verification_passed']


def run_guarded(function, input_path, output_dir, filename):
    """Malformed inputs also leave a diagnostic and return a failing status."""
    try:
        return function(input_path, output_dir)
    except Exception as exc:
        return save_summary(
            {'input': str(input_path), 'error': f'{type(exc).__name__}: {exc}'},
            {'audit_completed': {'passed': False}}, Path(output_dir)/filename)
