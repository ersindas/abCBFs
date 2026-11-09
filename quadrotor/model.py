from safety import ASIF
from control import Control
from dynamics import Dynamics
from estimator import Estimator


class Simulation(ASIF, Control, Dynamics, Estimator):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        vanilla_flag=True,
        robust_flag=False,
        skip_backup_check=False,
    ) -> None:

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.vanilla_flag = vanilla_flag
        self.robust_flag = robust_flag
        self.skip_backup_check = skip_backup_check

        self.setupControl()
        self.setupDynamics()
        self.setupASIF()
        self.setupEstimator()
        self.checkValidBackup()
