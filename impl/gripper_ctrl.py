import numpy as np

class GripperCtrl:

    def __init__(self, tau_max):
        
        # Remember maximum force
        self.tau_max = tau_max

    def reset(self):
        pass

    def open_to(self, alpha):
        
        assert (0 <= alpha).all() and (alpha <= 1).all(), f"Commanded opening state ({alpha}) outside allowed range [0, 1]"

        return alpha * self.tau_max