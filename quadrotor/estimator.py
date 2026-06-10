"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import numpy as np


class Estimator:
    def setupEstimator(
        self,
    ) -> None:

        # Parameter bounds
        # theta = [c_x, c_z, delta_g, delta_m, delta_J, delta_l]
        self.theta_hat_low = np.array([-.1,-.1,-.1,-.1,-.01,-.005])
        self.theta_hat_high = np.array([.1,.1,.1,.1,.01,.005])

        # Initialize parameter estimates
        self.theta_hat = (self.theta_hat_low + self.theta_hat_high) / 2

    # TODO Put DREM estimator here and also the flow bounds 