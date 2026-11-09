"""Position projection of the smooth safe set used by the quadrotor model."""
import numpy as np


def saved_barrier_kappa(*datasets, legacy_default=10.):
    """Read matching scalar smoothing values; older archives used kappa=10."""
    values = []
    for data in datasets:
        raw = np.asarray(data['kappa'] if 'kappa' in data else legacy_default)
        if raw.shape != () or not np.isfinite(raw) or float(raw) <= 0.:
            raise ValueError('Saved kappa must be a finite positive scalar.')
        values.append(float(raw))
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError('The plotted runs must use the same safety smoothing kappa.')
    return values[0]


def softmin_diagonal_offset(kappa, constant_barrier):
    """Exact equal-coordinate clearance beyond the two hard position bounds."""
    if not (np.isfinite(kappa) and kappa > 0. and
            np.isfinite(constant_barrier) and constant_barrier > 0.):
        raise ValueError('kappa and the remaining maximum barrier must be positive.')
    log_cap = np.log(-np.expm1(-kappa * constant_barrier))
    return (np.log(2.) - log_cap) / kappa


def projected_softmin_boundary(x_max, z_max, samples=1200, *, px_min=1.,
                              pz_min=1., constant_barrier=(np.pi / 3.) ** 2, kappa=10.):
    """Contour of two position exponentials plus the remaining minimum term.

    Work in log coordinates so distant viewport edges do not underflow when
    kappa is large. The curve follows the true soft-min contour, not a rounded
    rectangle chosen for appearance. Vertices include its diagonal crossing.
    """
    softmin_diagonal_offset(kappa, constant_barrier)  # validate model values
    if not np.all(np.isfinite([x_max, z_max, px_min, pz_min])) or samples < 3:
        raise ValueError('Viewport coordinates must be finite and samples at least three.')
    log_cap = np.log(-np.expm1(-kappa * constant_barrier))
    log_qz = -kappa * (z_max - pz_min) - log_cap
    log_qx = -kappa * (x_max - px_min) - log_cap
    if np.logaddexp(log_qz, log_qx) >= 0.:
        raise ValueError('Viewport must contain a nonempty projected safe region.')
    lower = log_qz - np.log(-np.expm1(log_qz))
    upper = np.log(-np.expm1(log_qx)) - log_qx
    ratio = np.linspace(lower, upper, samples)
    # The change of curvature is concentrated near ratio=0. Keep it resolved
    # even when the remaining curve is almost parallel to a hard boundary.
    detail_lower, detail_upper = max(lower, -20.), min(upper, 20.)
    detailed = (np.linspace(detail_lower, detail_upper, samples)
                if detail_lower <= detail_upper else np.array([]))
    ratio = np.unique(np.concatenate([ratio, detailed,
                                     [0.] if lower <= 0. <= upper else []]))
    px = px_min - (log_cap - np.logaddexp(0., ratio)) / kappa
    pz = pz_min - (log_cap - np.logaddexp(0., -ratio)) / kappa
    return np.column_stack([px, pz])


def projected_safe_boundary(x_max, z_max, samples=1200, *, px_min=1.,
                            pz_min=1., theta_max=np.pi / 3., kappa=10.):
    """Return the projected h=0 contour from the top to the right viewport edge.

    Projection eliminates pitch by choosing theta=0, which maximizes its
    barrier. Therefore the boundary satisfies
    exp(-kappa*(px-px_min)) + exp(-kappa*(pz-pz_min))
        = 1 - exp(-kappa*theta_max**2).
    The projection does not assert that every pitch at these positions is safe.
    """
    return projected_softmin_boundary(x_max, z_max, samples, px_min=px_min,
                                      pz_min=pz_min, constant_barrier=theta_max ** 2,
                                      kappa=kappa)


def projected_terminal_boundary(x_max, z_max, samples=1200, *, kappa=10.):
    """Closure of the smooth terminal position projection (pitch margin .05).

    The pitch barrier is maximized at the reference attitude and zero angular
    rate; the two velocity exponential terms approach zero for large positive
    velocities. This is the terminal surrogate used by the implementation.
    """
    return projected_softmin_boundary(x_max, z_max, samples, px_min=2., pz_min=2.,
                                      constant_barrier=.05, kappa=kappa)


def projected_safe_polygon(x_max, z_max, **kwargs):
    """Closed-region vertices, with the upper-right viewport corner appended."""
    return np.vstack([projected_safe_boundary(x_max, z_max, **kwargs),
                      [x_max, z_max]])
