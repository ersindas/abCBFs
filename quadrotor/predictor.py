"""Differentiable, componentwise backup prediction certificate.

Analytic bounds are conditional on the robust unsaturation guard along the
whole prediction. Arrays evaluate its numerical approximation on the grid.
No sampling certificate or validated integrator is implied by this module.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
jax.config.update('jax_enable_x64', True)

class StructuredBackupPredictor:

    def __init__(self, P, *, dtau=0.02, n_steps=50, kappa=10.0, theta_low=None, theta_high=None, F=20.0, Mmax=20.0, g=9.81, minv=1.0, Jinv=4.0, th_r=np.deg2rad(25), kp=4.0, kd=2.0, px_min=1.0, pz_min=1.0, rx=1.0, rz=1.0, theta_max=np.deg2rad(60), rho_th=0.05, smooth=1e-06):
        low = np.asarray(
            theta_low if theta_low is not None else [-0.1, -0.1, -0.25, -0.35, -0.01, -0.005],
            dtype=float,
        )
        high = np.asarray(
            theta_high if theta_high is not None else [0.1, 0.1, 0.25, 0.35, 0.01, 0.005],
            dtype=float,
        )
        Pn = np.asarray(P, dtype=float)
        assert Pn.shape == (2, 2) and np.allclose(Pn, Pn.T)
        ev = np.linalg.eigvalsh(Pn)
        if ev[0] <= 0:
            raise ValueError('Pitch P must be positive definite.')
        K = np.array([kp, kd])
        Pi = np.linalg.inv(Pn)
        qvals = []
        for a in (Jinv + low[4], Jinv + high[4]):
            A = np.array([[0.0, 1.0], [-a * kp, -a * kd]])
            qvals.append(np.linalg.eigvalsh(-A.T @ Pn - Pn @ A)[0])
        q_raw = min(qvals)
        if q_raw <= 0:
            raise ValueError('Pitch P fails common quadratic contraction.')
        q = 0.99 * q_raw
        self.certificate = dict(
            q_raw=float(q_raw),
            q=float(q),
            decay=float(q / (2 * ev[-1])),
            beta=float(np.sqrt(Pn[1, 1])),
            angle_gain=float(np.sqrt(Pi[0, 0])),
            moment_gain=float(np.sqrt(K @ Pi @ K)),
            parameter_low=low.tolist(),
            parameter_high=high.tolist(),
            dtau=float(dtau),
            n_steps=int(n_steps),
            smooth=float(smooth),
        )
        self.n_steps = n_steps
        self.dtau = dtau
        self.P = Pn
        Pj = jnp.asarray(Pn)
        Kj = jnp.asarray(K)
        decay = self.certificate['decay']
        beta = self.certificate['beta']
        angle_gain = self.certificate['angle_gain']
        moment_gain = self.certificate['moment_gain']
        forcing_radius = beta * F * max(abs(low[5]), abs(high[5])) / decay
        self.certificate['forcing_radius'] = float(forcing_radius)
        am_max = max(abs(minv + low[3]), abs(minv + high[3]))
        growth_x = -low[0]
        growth_z = -low[1]

        def smooth_abs(z):
            return jnp.sqrt(z * z + smooth * smooth)

        def rhs(z, that, rho):
            x = z[:6]
            r, rvx, rvz, rpx, rpz = z[6:]
            px, pz, th, vx, vz, om = x
            raw_M = kp * (th - th_r) + kd * om
            M = jnp.clip(raw_M, -Mmax, Mmax)
            cx, cz, dg, dm, dJ, dl = that
            dx = jnp.array([
                vx,
                vz,
                om,
                -cx * vx + (minv + dm) * F * jnp.sin(th),
                -cz * vz - g - dg + (minv + dm) * F * jnp.cos(th),
                -(Jinv + dJ) * M + F * dl,
            ])
            dr = -decay * r + beta * (rho[4] * smooth_abs(raw_M) + F * rho[5])
            rth = angle_gain * r
            drvx = growth_x * rvx + rho[0] * smooth_abs(vx) + F * am_max * rth + F * rho[3] * smooth_abs(jnp.sin(th))
            drvz = growth_z * rvz + rho[1] * smooth_abs(vz) + rho[2] + F * am_max * rth + F * rho[3] * smooth_abs(jnp.cos(th))
            return jnp.concatenate([dx, jnp.array([dr, drvx, drvz, rvx, rvz])])

        def prediction(x, that, rho):

            def advance(z, _):
                a = rhs(z, that, rho)
                b = rhs(z + 0.5 * dtau * a, that, rho)
                c = rhs(z + 0.5 * dtau * b, that, rho)
                d = rhs(z + dtau * c, that, rho)
                z1 = z + dtau / 6 * (a + 2 * b + 2 * c + d)
                return (z1, z1)
            init = jnp.concatenate([x, jnp.zeros(5)])
            _, traj = jax.lax.scan(advance, init, None, length=n_steps)
            aug = jnp.vstack([init, traj])
            states = aug[:, :6]
            rad = aug[:, 6:]
            r, rvx, rvz, rpx, rpz = rad.T
            th = states[:, 2]
            rth = angle_gain * r
            running_comp = jnp.stack(
                [
                    states[:, 0] - px_min - rpx,
                    states[:, 1] - pz_min - rpz,
                    theta_max ** 2 - th ** 2 - 2 * smooth_abs(th) * rth - rth ** 2,
                ],
                axis=1,
            )
            y = jnp.stack([th - th_r, states[:, 5]], axis=1)
            V = jnp.einsum('ni,ij,nj->n', y, Pj, y)
            pitch_lower = rho_th - V - 2 * jnp.sqrt(V + smooth * smooth) * r - r * r
            backup_comp = jnp.stack(
                [
                    states[:, 0] - px_min - rx - rpx,
                    states[:, 1] - pz_min - rz - rpz,
                    pitch_lower,
                    states[:, 3] - rvx,
                    states[:, 4] - rvz,
                ],
                axis=1,
            )
            safe = -logsumexp(-kappa * running_comp, axis=1) / kappa
            backup = -logsumexp(-kappa * backup_comp, axis=1) / kappa
            hbars = jnp.concatenate([safe, backup[-1:]])
            moment_bound = smooth_abs(y @ Kj) + moment_gain * r
            all_time_moment_bound = moment_gain * jnp.maximum(
                jnp.sqrt(jnp.maximum(V[0], 0.0)),
                forcing_radius,
            )
            return dict(
                states=states,
                radii=rad,
                hbars=hbars,
                running_components=running_comp,
                backup_components=backup_comp,
                backup=backup,
                moment_bound=moment_bound,
                moment_margin=Mmax - moment_bound,
                all_time_moment_bound=all_time_moment_bound,
            )

        def barriers(x, that, rho):
            return prediction(x, that, rho)['hbars']
        self.evaluate = jax.jit(prediction)
        self.barriers = jax.jit(barriers)
        jac = jax.jacfwd(barriers, argnums=(0, 1, 2))

        @jax.jit
        def values_and_jacobians(x, that, rho):
            values = barriers(x, that, rho)
            Jx, Jtheta, Jrho = jac(x, that, rho)
            return (values, Jx, Jtheta, Jrho)
        self.values_and_jacobians = values_and_jacobians
