import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        ### TODO: Please implement me
        # u = None
        u = self.kp * q_d - q + self.kd * q_d_dot - q_dot
        return u
