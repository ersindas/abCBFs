import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import jacfwd
import jax
from functools import partial

jax.config.update("jax_enable_x64", True)


def f_x_nom(x, lenx, g):
    f = jnp.zeros(lenx)
    f = f.at[:3].set(x[3:6])
    f = f.at[4].set(-g)
    return f

def g_x_nom(x, lenx, m, Jinv):
    g = jnp.zeros((lenx, 2))
    g = g.at[3:, :].set(
        jnp.array(
            [
                [jnp.sin(x[2]) / m, 0.0],
                [jnp.cos(x[2]) / m, 0.0],
                [0.0, -Jinv],
            ]
        )
    )
    return g

def backupControl_pos_jax(x, F_max, M_max, params):
    th = x[2]
    thd = x[5]
    th_r = params["th_r"]
    Kp_th = params["Kp_th"]
    Kd_th = params["Kd_th"]

    M = Kp_th * (th - th_r) + Kd_th * thd
    M = jnp.clip(M, -M_max, M_max)

    u_b = jnp.array([F_max, M])
    return u_b

def fb_vanilla(
    x,
    lenx,
    g,
    m,
    Jinv,
    backupParams,
    F_max,
    M_max,
):
    return f_x_nom(x, lenx, g) + g_x_nom(x, lenx, m, Jinv) @ backupControl_pos_jax(
        x, F_max, M_max, backupParams
    )

############### adaptive backup flow ###############

def psi_b_jax(x, F_max, M_max, backupParams):
    # regression matrix psi_b(x), columns ordered as
    # theta = [c_x, c_z, delta_g, delta_m, delta_J, delta_l]
    u_b = backupControl_pos_jax(x, F_max, M_max, backupParams)
    F = u_b[0]
    M = u_b[1]
    th = x[2]
    vx = x[3]
    vz = x[4]
    psi = jnp.zeros((6, 6))
    psi = psi.at[3, 0].set(-vx)                  # c_x
    psi = psi.at[4, 1].set(-vz)                  # c_z
    psi = psi.at[4, 2].set(-1.0)                 # delta_g
    psi = psi.at[3, 3].set(jnp.sin(th) * F)      # delta_m
    psi = psi.at[4, 3].set(jnp.cos(th) * F)
    psi = psi.at[5, 4].set(-M)                   # delta_J
    psi = psi.at[5, 5].set(F)                    # delta_l
    return psi

def fb_adaptive(x, theta_hat, lenx, g, m, Jinv, backupParams, F_max, M_max):
    # estimated closed-loop backup dynamics f_b(x, theta_hat)
    return (
        f_x_nom(x, lenx, g)
        + g_x_nom(x, lenx, m, Jinv)
        @ backupControl_pos_jax(x, F_max, M_max, backupParams)
        + psi_b_jax(x, F_max, M_max, backupParams) @ theta_hat
    )

def _smooth_col_norms(z, F_max, M_max, backupParams, smooth_s, Tmat):
    # ||T psi_b,i(z)||_{2,s}, smoothed column norms in the same weighted
    # norm ||.||_T = ||T.||_2 that Lb is certified in
    psi = psi_b_jax(z, F_max, M_max, backupParams)        # (6,6)
    Tpsi = Tmat @ psi                                       # T applied to every column
    return jnp.sqrt(jnp.sum(Tpsi**2, axis=0) + smooth_s**2)  # (6,)

def d_b_jax(z, rho, Tmat, F_max, M_max, backupParams, smooth_s):
    # d_b(z,rho) = sum_i ||T psi_b,i(z)||_{2,s} rho_i, in ||.||_T (Tmat=I -> plain norm)
    return jnp.sum(rho * _smooth_col_norms(z, F_max, M_max, backupParams, smooth_s, Tmat))

def _adaptive_rollout(x0, theta_hat, rho, rho_dot, Lb, Tmat, n_steps, dtau,
                       fb, fb_jac, psi_fn, db_fn, db_grad_z_fn, lenx):
    """
    fixed-step RK4 rollout of the augmented system
        z'     = f_b(z, theta_hat)
        Phi'   = J(z) Phi,                       Phi(0)   = I    state sensitivity
        Gamma' = J(z) Gamma + psi(z),             Gamma(0) = 0    estimate sensitivity
        delta' = Lb delta + d_b(z,rho),           delta(0) = 0    Gronwall bound, in ||.||_T
        Xi'    = Lb Xi + (d d_b/dz)(z,rho) Phi,   Xi(0)    = 0    = d delta/dx
        Q'     = Lb Q + d_b(z,rho_dot),           Q(0)     = 0    = d delta/dt

    Lb must be the log-norm of f_b's Jacobian in the same ||.||_T (mu(Tmat @ Jcl @
    Tmat^-1)); d_b uses Tmat-weighted column norms to match, so delta_max bounds
    ||e(tau)||_T = ||Tmat e(tau)||_2 consistently throughout.
    """
    def aug_rhs(carry):
        z, Phi, Gam, delta, Xi, Q = carry
        J = fb_jac(z, theta_hat)
        db = db_fn(z, rho, Tmat)
        db_rate = db_fn(z, rho_dot, Tmat)
        dgrad = db_grad_z_fn(z, rho, Tmat)
        return (
            fb(z, theta_hat),
            J @ Phi,
            J @ Gam + psi_fn(z),
            Lb * delta + db,
            Lb * Xi + dgrad @ Phi,
            Lb * Q + db_rate,
        )

    def rk4(carry, _):
        z, Phi, Gam, delta, Xi, Q = carry
        k1 = aug_rhs((z, Phi, Gam, delta, Xi, Q))
        k2 = aug_rhs((z + 0.5 * dtau * k1[0], Phi + 0.5 * dtau * k1[1],
                      Gam + 0.5 * dtau * k1[2], delta + 0.5 * dtau * k1[3],
                      Xi + 0.5 * dtau * k1[4], Q + 0.5 * dtau * k1[5]))
        k3 = aug_rhs((z + 0.5 * dtau * k2[0], Phi + 0.5 * dtau * k2[1],
                      Gam + 0.5 * dtau * k2[2], delta + 0.5 * dtau * k2[3],
                      Xi + 0.5 * dtau * k2[4], Q + 0.5 * dtau * k2[5]))
        k4 = aug_rhs((z + dtau * k3[0], Phi + dtau * k3[1],
                      Gam + dtau * k3[2], delta + dtau * k3[3],
                      Xi + dtau * k3[4], Q + dtau * k3[5]))
        z_n = z + dtau / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Phi_n = Phi + dtau / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Gam_n = Gam + dtau / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        delta_n = delta + dtau / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        Xi_n = Xi + dtau / 6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        Q_n = Q + dtau / 6 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])
        new_carry = (z_n, Phi_n, Gam_n, delta_n, Xi_n, Q_n)
        return new_carry, new_carry

    init = (x0, jnp.eye(lenx), jnp.zeros((lenx, 6)), 0.0, jnp.zeros(lenx), 0.0)
    _, traj = jax.lax.scan(rk4, init, None, length=n_steps)
    zs = jnp.vstack([x0[None, :], traj[0]])
    Phis = jnp.concatenate([jnp.eye(lenx)[None], traj[1]], axis=0)
    Gams = jnp.concatenate([jnp.zeros((1, lenx, 6)), traj[2]], axis=0)
    deltas = jnp.concatenate([jnp.zeros((1,)), traj[3]], axis=0)
    Xis = jnp.concatenate([jnp.zeros((1, lenx)), traj[4]], axis=0)
    Qs = jnp.concatenate([jnp.zeros((1,)), traj[5]], axis=0)
    return zs, Phis, Gams, deltas, Xis, Qs


class Dynamics:
    def setupDynamics(self):

        # nominal constants
        self.g0 = 9.81  # [m/s^2]
        self.m0 = 1  # [kg]
        self.J0 = 0.25  # [kgm^2]
        self.Jinv = 1 / self.J0

        # true parameters (inside Theta of the estimator)
        # theta = [c_x, c_z, delta_g, delta_m, delta_J, delta_l]
        self.c_x = 0.08
        self.c_z = 0.08
        self.delta_g = 0.22    # extra specific gravity the nominal model ignores
        self.delta_m = -0.32   # heavy: true mass m = 1/(1/m0 + delta_m) = 1.47 kg
        self.delta_J = 0.008
        self.delta_l = 0.003
        # true theta sits well inside the estimator box (delta_m is 91% of the
        # box half-width); weakens vertical thrust authority ~2.8x vs nominal,
        # which is why the vanilla bCBF crashes through the floor but the
        # robust-adaptive one doesn't
        self.bTheta = np.array([self.c_x, self.c_z, self.delta_g, self.delta_m, self.delta_J, self.delta_l])

        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        self.del_t = 0.01  # [s]
        self.total_steps = int(4.0 / self.del_t) + 1
        self.curr_step = 0

        # [x, z, theta, xdot, zdot, thetadot] -- start descending fast toward
        # the wall/floor corner so the floor constraint stays active
        self.x0 = np.array([7.0, 3.0, 0.0, -7.5, -0.8, 0.0])
        self.lenx = len(self.x0)

        self._dx_backup = np.zeros(self.lenx + self.lenx**2)

        self._stm_start = self.lenx
        self._stm_end = self.lenx + self.lenx**2
        self._stm_shape = (self.lenx, self.lenx)

        self._fb_backup_kwargs = dict(
            lenx=self.lenx,
            g=self.g0,
            m=self.m0,
            Jinv=self.Jinv,
            backupParams=self.backupParams,
            F_max=self.F_max,
            M_max=self.M_max,
        )

        self.kb_vanilla = backupControl_pos_jax
        self.jit_fb_vanilla = jax.jit(partial(fb_vanilla, **self._fb_backup_kwargs))
        self.fb_vanilla_jac_jax_new = jax.jit(
            jacfwd(partial(fb_vanilla, **self._fb_backup_kwargs), argnums=0)
        )

        # adaptive backup flow (jitted RK4 rollout)
        fb_ad = partial(fb_adaptive, **self._fb_backup_kwargs)
        fb_ad_jac = jacfwd(fb_ad, argnums=0)
        psi_fn = partial(
            psi_b_jax,
            F_max=self.F_max,
            M_max=self.M_max,
            backupParams=self.backupParams,
        )
        self.smooth_s = 0.01   # ||.||_{2,s} regularization
        db_kwargs = dict(F_max=self.F_max, M_max=self.M_max,
                          backupParams=self.backupParams, smooth_s=self.smooth_s)
        db_fn = partial(d_b_jax, **db_kwargs)
        db_grad_z_fn = jax.grad(d_b_jax, argnums=0)
        db_grad_z_fn = partial(db_grad_z_fn, **db_kwargs)
        self.jit_fb_adaptive = jax.jit(fb_ad)
        self.jit_fb_adaptive_jac = jax.jit(fb_ad_jac)
        self.jit_psi_b = jax.jit(psi_fn)
        self._adaptive_rollout_jit = jax.jit(
            partial(
                _adaptive_rollout,
                fb=fb_ad,
                fb_jac=fb_ad_jac,
                psi_fn=psi_fn,
                db_fn=db_fn,
                db_grad_z_fn=db_grad_z_fn,
                lenx=self.lenx,
            ),
            static_argnames=("n_steps",),
        )

    def adaptiveBackupFlow(self, x, theta_hat, rho, rho_dot, Lb, Tmat, n_steps, dtau):
        # estimated backup flow with sensitivities, plus the (x,t)-dependent
        # Gronwall bound delta_max and its sensitivities Xi = d(delta_max)/dx,
        # Q = d(delta_max)/dt, all consistent in the weighted norm ||.||_T
        # (Tmat=I recovers the plain euclidean norm)
        zs, Phis, Gams, deltas, Xis, Qs = self._adaptive_rollout_jit(
            jnp.asarray(x), jnp.asarray(theta_hat), jnp.asarray(rho),
            jnp.asarray(rho_dot), float(Lb), jnp.asarray(Tmat),
            n_steps=n_steps, dtau=dtau,
        )
        return (np.asarray(zs), np.asarray(Phis), np.asarray(Gams),
                np.asarray(deltas), np.asarray(Xis), np.asarray(Qs))

    def f_x_nominal(self, x):
        f = np.zeros_like(x)
        f[:3] = x[3:6]
        f[4] = -self.g0
        return f

    def g_x_nominal(self, x):
        g = np.zeros((self.lenx, 2))
        g[3:, :] = np.array(
            [[np.sin(x[2]) / self.m0, 0], [np.cos(x[2]) / self.m0, 0], [0, -self.Jinv]]
        )
        return g

    def f_x_hat(self, x, theta_hat):
        # estimated drift f(x) + F(x) theta_f_hat
        f = self.f_x_nominal(x)
        f[3] += -theta_hat[0] * x[3]
        f[4] += -theta_hat[1] * x[4] - theta_hat[2]
        return f

    def g_x_hat(self, x, theta_hat):
        # estimated actuation matrix g(x) + sum_j theta_g_hat_j G_j(x)
        am_hat = 1 / self.m0 + theta_hat[3]
        aJ_hat = 1 / self.J0 + theta_hat[4]
        g = np.zeros((self.lenx, 2))
        g[3:, :] = np.array(
            [
                [np.sin(x[2]) * am_hat, 0],
                [np.cos(x[2]) * am_hat, 0],
                [theta_hat[5], -aJ_hat],
            ]
        )
        return g

    ###################### backup #########################

    def propMainBackup_vanilla_jax(self, t, x):
        # propagation function for backup dynamics + STM
        dx = self._dx_backup
        dx[: self.lenx] = self.jit_fb_vanilla(x[: self.lenx])

        F = np.asarray(self.fb_vanilla_jac_jax_new(x[: self.lenx]))

        STM = x[self._stm_start : self._stm_end].reshape(self._stm_shape)
        dSTM = F @ STM
        dx[self._stm_start : self._stm_end] = dSTM.ravel()

        return dx

    ###################### main sim #########################

    def propState(self, t, x, u):
        # true dynamics, x = [px pz theta vx vz omega]
        dx = np.zeros_like(x)
        dx[0:3] = x[3:]
        dx[3] = (np.sin(x[2])/self.m0)*u[0] - self.c_x*x[3] + self.delta_m*np.sin(x[2])*u[0]
        dx[4] = -self.g0 - self.c_z*x[4] - self.delta_g + (np.cos(x[2])/self.m0)*u[0] +self.delta_m*np.cos(x[2])*u[0]
        dx[5] = -u[1]/self.J0 - self.delta_J*u[1] + self.delta_l*u[0]

        return dx

    def integrateState(self, x, u, t_curr, t_step, options):
        t_int = (t_curr, t_curr + t_step)
        soltn = solve_ivp(
            lambda t, x: self.propState(t, x, u),
            t_int,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x
