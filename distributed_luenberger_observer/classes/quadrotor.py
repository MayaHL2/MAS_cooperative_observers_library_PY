import numpy as np

class Quadrotor:
    g = 9.81
    def __init__(self, mass, distance_cg_motor, J, lift_factor, drag_factor):
        self.A = np.array(
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, self.g, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -self.g, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )

        self.B = np.array(
            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/mass, 0, 0, 0],
            [0, 1/J[0], 0, 0],
            [0, 0, 1/J[1], 0],
            [0, 0, 0, 1/J[2]]]
        )

        self.C = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
        )

        self.l = distance_cg_motor
        self.b = lift_factor
        self.d = drag_factor

    def get_state_space(self):
        return self.A, self.B, self.C

    def rotation_speeds_2_inputs(self, Omega):
        self.inputs = np.dot(np.array(
            [[self.b, self.b, self.b, self.b],
            [0, -self.b*self.l, 0, self.b*self.l],
            [-self.b*self.l, 0, self.b*self.l, 0],
            [-self.d, self.d, -self.d, self.d]]
        ), np.array(Omega)**2)
        return self.inputs

    def inputs_2_rotation_speeds(self, inputs):
        self.Omega = (np.dot(np.linalg.inv(np.array(
            [[self.b, self.b, self.b, self.b],
            [0, -self.b*self.l, 0, self.b*self.l],
            [-self.b*self.l, 0, self.b*self.l, 0],
            [-self.d, self.d, -self.d, self.d]]
        )), inputs))**(1/2)
        return self.Omega