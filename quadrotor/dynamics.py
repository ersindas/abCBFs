import numpy as np
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
        self.delta_m = -0.32   # Updated by experiment_config before simulation.
        self.delta_J = 0.008
        self.delta_l = 0.003
        self.bTheta = np.array([self.c_x, self.c_z, self.delta_g, self.delta_m, self.delta_J, self.delta_l])

        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        self.del_t = 0.01  # [s]
        self.curr_step = 0

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
