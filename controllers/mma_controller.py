import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManiuplatorModel(Tp, m3 = 0.1, r3 = 0.05),
                       ManiuplatorModel(Tp, m3 = 0.01, r3 = 0.01),
                       ManiuplatorModel(Tp, m3 = 1.0, r3 = 0.3)]
        self.i = 0
        self.u = np.zeros((2, 1))

    def choose_model(self, x):
        q1, q2, q1_dot, q2_dot = x
        prev_error = np.inf
        for idx, model in enumerate(self.models):
            y = np.dot(model.M(x), self.u) + np.dot(model.C(x), [[q1_dot],[q2_dot]])
            error = np.sum(np.abs([[q1],[q2]]-y))
            if error < prev_error:
                prev_error = error
                self.i = idx

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        # v = q_r_ddot # TODO: add feedback

        K_d = np.array([[20, 0], [0, 30]])
        K_p = np.array([[45, 0], [0, 60]])

        v = q_r_ddot - K_d @ (q_dot - q_r_dot) - K_p @ (q - q_r) 

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
