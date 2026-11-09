"""Explicit lifted robust QP, with no objective penalty on dual variables."""

import time
import numpy as np
from scipy import sparse
import clarabel


def error_polytope(theta_hat, rho, theta_low, theta_high):
    """Return the manuscript's symmetric 2N-row certified error box.

    The estimate and prior arguments retain the existing caller interface.
    The prior box is not intersected with the certified error box.
    """
    theta_hat, rho, theta_low, theta_high = map(
        lambda v: np.asarray(v, dtype=float),
        (theta_hat, rho, theta_low, theta_high),
    )
    n = theta_hat.size
    if rho.shape != (n,) or not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("The certified radii must be finite and nonnegative.")
    A = np.vstack((np.eye(n), -np.eye(n)))
    b = np.concatenate((rho, rho))
    lower = -rho.copy()
    upper = rho.copy()
    return A, b, lower, upper


def box_minimum(c, lower, upper):
    """Exact support minimum, used only for independent solution checks."""
    c = np.asarray(c, dtype=float)
    return np.sum(np.minimum(c * lower, c * upper), axis=-1)


def solve_lifted_qp(a0, au, c0, cu, A, b, u_low, u_high, u_des,
                    *, tolerance=1e-9, check_tolerance=1e-6):
    """Solve min ||u-u_des||^2 over w=(u, mu_0,...,mu_G-1).

    Each group imposes a0+au*u+b.T*mu>=0, A.T*mu=c0+cu*u,
    and mu<=0. Clarabel's internal slack representation is only the exact
    conic encoding of these inequalities; no relaxed safety constraint or
    extra objective regularization is introduced.

    A failed solve returns success=False. Only classification
    'primal_infeasible' includes a checked numerical Farkas certificate.
    """
    tic = time.perf_counter()
    a0, au, c0, cu, A, b, u_low, u_high, u_des = map(
        lambda v: np.asarray(v, dtype=float),
        (a0, au, c0, cu, A, b, u_low, u_high, u_des),
    )
    groups, n, m = cu.shape
    rows_per_dual = A.shape[0]
    if (a0.shape != (groups,) or au.shape != (groups, m)
            or c0.shape != (groups, n) or A.shape[1] != n
            or b.shape != (rows_per_dual,)):
        raise ValueError("Inconsistent lifted QP coefficient dimensions.")
    if not all(np.all(np.isfinite(v)) for v in
               (a0, au, c0, cu, A, b, u_low, u_high, u_des)):
        raise ValueError("Nonfinite lifted QP coefficient.")

    ndual = groups * rows_per_dual
    nvar = m + ndual
    neq = groups * n
    equalities = sparse.hstack((
        sparse.csc_matrix(-cu.reshape(neq, m)),
        sparse.kron(sparse.eye(groups, format="csc"),
                    sparse.csc_matrix(A.T), format="csc"),
    ), format="csc")
    input_rows = sparse.hstack((
        sparse.csc_matrix(np.vstack((np.eye(m), -np.eye(m)))),
        sparse.csc_matrix((2*m, ndual)),
    ), format="csc")
    nonpositive_duals = sparse.hstack((
        sparse.csc_matrix((ndual, m)), sparse.eye(ndual, format="csc"),
    ), format="csc")
    barriers = sparse.hstack((
        sparse.csc_matrix(-au),
        sparse.kron(sparse.eye(groups, format="csc"),
                    sparse.csc_matrix(-b.reshape(1, -1)), format="csc"),
    ), format="csc")
    conic_A = sparse.vstack((equalities, input_rows, nonpositive_duals,
                            barriers), format="csc")
    conic_b = np.concatenate((c0.ravel(), u_high, -u_low,
                              np.zeros(ndual), a0))
    # Positive row scaling changes neither w nor the feasible set. In
    # particular, large cubic-alpha constants otherwise dominate the
    # infeasibility certificate before its stationarity residual is small.
    row_norms = np.asarray(abs(conic_A).max(axis=1).toarray()).ravel()
    row_scale = 1.0/np.maximum.reduce((np.ones_like(conic_b),
                                     np.abs(conic_b), row_norms))
    scaled_A = sparse.diags(row_scale, format="csc") @ conic_A
    scaled_b = row_scale * conic_b
    P = sparse.diags(np.r_[np.full(m, 2.0), np.zeros(ndual)], format="csc")
    q = np.r_[-2*u_des, np.zeros(ndual)]
    cones = [clarabel.ZeroConeT(neq),
             clarabel.NonnegativeConeT(2*m + ndual + groups)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = tolerance
    settings.tol_gap_rel = tolerance
    settings.tol_feas = tolerance
    settings.tol_infeas_abs = tolerance
    settings.tol_infeas_rel = tolerance
    settings.tol_ktratio = 1e-12
    settings.max_threads = 1
    result = dict(
        success=False, classification="numerical_failure", status="NotRun",
        w=None, mu=None, u=None, a0=a0, au=au, c0=c0, cu=cu, A=A, b=b,
        u_low=u_low, u_high=u_high, u_des=u_des,
        n_variables=nvar, n_dual_groups=groups,
        n_equalities=neq, n_inequalities=2*m+ndual+groups,
        solver="clarabel", solver_version=clarabel.__version__,
        solver_tolerance=tolerance, check_tolerance=check_tolerance,
        constraint_row_scale=row_scale,
        native_solver_status="NotRun", solver_time=0.0, setup_time=0.0,
        solve_time=0.0, presolve_time=0.0, iterations=0,
    )
    result["assembly_time"] = time.perf_counter() - tic
    presolve_start = time.perf_counter()
    # If zero error is admissible, each robust row requires a0+au*u>=0.
    # A negative maximum over the input box has an explicit Farkas ray for
    # the FULL lifted problem. This exact presolve neither changes the
    # optimization problem nor interprets a numerical failure as infeasibility.
    if np.all(b >= 0):
        row_upper_bounds = a0 + np.sum(np.maximum(au*u_low, au*u_high), axis=1)
        g = int(np.argmin(row_upper_bounds))
        if row_upper_bounds[g] < 0:
            certificate = np.zeros(conic_b.size)
            certificate[neq:neq+m] = np.maximum(au[g], 0)
            certificate[neq+m:neq+2*m] = np.maximum(-au[g], 0)
            offset = neq + 2*m + g*rows_per_dual
            certificate[offset:offset+rows_per_dual] = b
            certificate[neq+2*m+ndual+g] = 1.0
            scale = max(1.0, np.linalg.norm(certificate, ord=np.inf))
            stationarity = np.max(np.abs(conic_A.T @ certificate))/scale
            cone_violation = max(0.0, -np.min(certificate[neq:]))/scale
            separation = float(conic_b @ certificate)/scale
            if (stationarity <= check_tolerance
                    and cone_violation <= check_tolerance
                    and separation < -check_tolerance):
                result.update(
                    status="PrimalInfeasibleByCertificate",
                    classification="primal_infeasible",
                    infeasibility_certificate=certificate,
                    certificate_source="zero_error_constraint",
                    certificate_group=g,
                    certificate_row_upper_bound=float(row_upper_bounds[g]),
                    certificate_stationarity=stationarity,
                    certificate_cone_violation=cone_violation,
                    certificate_separation=separation,
                )
                result["presolve_time"] = time.perf_counter()-presolve_start
                return result
    result["presolve_time"] = time.perf_counter()-presolve_start
    tic = time.perf_counter()
    try:
        solver = clarabel.DefaultSolver(P, q, scaled_A, scaled_b, cones, settings)
        result["setup_time"] = time.perf_counter() - tic
        solve_start = time.perf_counter()
        sol = solver.solve()
        result["solve_time"] = time.perf_counter() - solve_start
    except Exception as exc:
        result["status"] = "SolverException"
        result["error"] = repr(exc)
        result["solver_time"] = time.perf_counter() - tic
        return result
    result.update(
        status=str(sol.status), solver_time=time.perf_counter()-tic,
        native_solver_status=str(sol.status),
        iterations=sol.iterations, primal_solver_residual=sol.r_prim,
        dual_solver_residual=sol.r_dual,
        objective_primal=sol.obj_val, objective_dual=sol.obj_val_dual,
    )
    if result["status"] == "PrimalInfeasible":
        # A_scaled=S*A: the original-coordinate certificate is S*z_scaled.
        # Primal variables, including every mu, are unchanged by row scaling.
        certificate = row_scale * np.asarray(sol.z, dtype=float)
        scale = max(1.0, np.linalg.norm(certificate, ord=np.inf))
        stationarity = np.max(np.abs(conic_A.T @ certificate))/scale
        cone_violation = max(0.0, -np.min(certificate[neq:]))/scale
        separation = float(conic_b @ certificate)/scale
        result.update(infeasibility_certificate=certificate,
                      certificate_source="clarabel",
                      certificate_stationarity=stationarity,
                      certificate_cone_violation=cone_violation,
                      certificate_separation=separation)
        if (np.isfinite(stationarity) and stationarity <= check_tolerance
                and cone_violation <= check_tolerance
                and separation < -check_tolerance):
            result["classification"] = "primal_infeasible"
        return result
    if result["status"] != "Solved":
        return result

    w = np.asarray(sol.x, dtype=float)
    u = w[:m]
    mu = w[m:].reshape(groups, rows_per_dual)
    eq_residual = mu @ A - c0 - np.einsum("gnm,m->gn", cu, u)
    dual_barriers = a0 + au @ u + mu @ b
    input_violation = max(0.0, np.max(u_low-u), np.max(u-u_high))
    mu_violation = max(0.0, np.max(mu))
    # Validate the dual solution against the symmetric box support minimum.
    if rows_per_dual != 2*n or not np.array_equal(
            A, np.vstack((np.eye(n), -np.eye(n)))):
        raise ValueError("Independent validation expects the two-block error box.")
    lower = -b[n:2*n]
    upper = b[:n]
    c = c0 + np.einsum("gnm,m->gn", cu, u)
    robust_barriers = a0 + au @ u + box_minimum(c, lower, upper)
    eq_scale = 1 + np.abs(c0) + np.abs(np.einsum("gnm,m->gn", cu, u))
    result.update(
        w=w, mu=mu, u=u, equality_residual_inf=np.max(np.abs(eq_residual)),
        equality_residual_scaled=np.max(np.abs(eq_residual)/eq_scale),
        mu_positive_violation=mu_violation, input_violation=input_violation,
        min_dual_barrier_residual=np.min(dual_barriers),
        min_robust_barrier_residual=np.min(robust_barriers),
        dual_barrier_residuals=dual_barriers,
        robust_barrier_residuals=robust_barriers,
        objective=float(np.sum((u-u_des)**2)),
        parameter_error_lower=lower, parameter_error_upper=upper,
    )
    if (np.all(np.isfinite(w))
            and result["equality_residual_scaled"] <= check_tolerance
            and input_violation <= check_tolerance
            and mu_violation <= check_tolerance
            and np.min(dual_barriers) >= -check_tolerance
            and np.min(robust_barriers) >= -check_tolerance):
        result["success"] = True
        result["classification"] = "optimal"
    else:
        result["status"] = "SolvedResidualCheckFailed"
    return result
