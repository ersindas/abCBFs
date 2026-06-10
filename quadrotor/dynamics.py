"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import jacfwd
import jax
from functools import partial

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

# TODO DO THE SAME THING FOR THE ADAPTIVE FLOW -- DEFINE IN THE SAME WAY AS ABOVE WHICH GIVES A BIG SPEED UP

class Dynamics:
    def setupDynamics(
        self,
    ) -> None:

        # Nominal constants
        self.g0 = 9.81  # [m/s^2]
        self.m0 = 1  # [kg]
        self.J0 = 0.25  # [kgm^2]
        self.Jinv = 1 / self.J0
 
        # True parameters # TODO Adjust these
        self.c_x = 0.0
        self.c_z = 0.0
        self.delta_g = 0.0
        self.delta_m = 0.0
        self.delta_J = 0.0
        self.delta_l = 0.0
        self.bTheta = np.array([self.c_x, self.c_z, self.delta_g, self.delta_m, self.delta_J, self.delta_l])

        # Integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Simulation data
        self.del_t = 0.01  # [s]
        # self.total_steps = int(3.5 / self.del_t) + 1
        # self.total_steps = 25
        self.total_steps = int(2.0 / self.del_t) + 1
        self.curr_step = 0

        # Initial conditions [x, z, theta, xdot, zdot, thetadot]
        self.x0 = np.array([5.5, 6.5, 0.1, 0, 0, 0])
        self.lenx = len(self.x0)

        # Constant
        self.sup_fcl = 1

        # Jax/jit
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
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = np.zeros_like(x)
        f[:3] = x[3:6]
        f[4] = -self.g0
        return f

    def g_x_nominal(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = np.zeros((self.lenx, 2))
        g[3:, :] = np.array(
            [[np.sin(x[2]) / self.m0, 0], [np.cos(x[2]) / self.m0, 0], [0, -self.Jinv]]
        )
        return g

    ###################### BACKUP #########################
    
    def propMainBackup_vanilla_jax(self, t, x):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        dx = self._dx_backup
        dx[: self.lenx] = self.jit_fb_vanilla(x[: self.lenx])

        F = np.asarray(self.fb_vanilla_jac_jax_new(x[: self.lenx]))

        STM = x[self._stm_start : self._stm_end].reshape(self._stm_shape)
        dSTM = F @ STM
        dx[self._stm_start : self._stm_end] = dSTM.ravel()

        return dx

    ###################### MAIN SIM #########################

    def propState(self, t, x, u):
        """
        Propagation function for dynamics. (TRUE DYNAMICS)
        x = [px pz theta vx vz omega]

        """ 
        dx = np.zeros_like(x)
        dx[0:3] = x[3:]
        dx[3] = (np.sin(x[2])/self.m0)*u[0] - self.c_x*x[3] + self.delta_m*np.sin(x[2])*u[0]
        dx[4] = -self.g0 - self.c_z*x[4] - self.delta_g + (np.cos(x[2])/self.m0)*u[0] +self.delta_m*np.cos(x[2])*u[0]
        dx[5] = -u[1]/self.J0 - self.delta_J*u[1] + self.delta_l*u[0]

        return dx

    def integrateState(self, x, u, t_curr, t_step, options):
        """
        Propagate state.

        """
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
