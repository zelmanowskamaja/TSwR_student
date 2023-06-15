import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.model = ManiuplatorModel(Tp)

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [self.b],
                      [0]])
        L = np.array([[3*p],
                      [3*p**2],
                      [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        # return NotImplementedError
        self.b = b
        B = np.array([[0],
                    [b],
                    [0]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, indx):
        ### TODO implement ADRC
        # return NotImplementedError
        state = self.eso.get_state()

        v = q_d_ddot + self.kd * (q_d_dot - state[1]) + self.kp * (q_d - x[0])
        u = (v - state[2]) / self.b

        self.eso.update(x[0], u)

        M = self.model.M([x[0], x[1], 0.0, 0.0])
        self.set_b(np.linalg.inv(M)[indx, indx])
        return u
