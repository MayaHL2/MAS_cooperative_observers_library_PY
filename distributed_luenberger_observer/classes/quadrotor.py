import numpy as np

class Quadrotor:
    """ This class defines mathematically the system of a quadrotor
    The system is defined using the following reference:
    https://ieeexplore.ieee.org/abstract/document/6915128/?casa_token=djSrqFN7KRcAAAAA:eptIjkYXaYX6QCypspMuiZGrqVrwgCQDAfQpZ4wHuxDcG39yWsxo7esStLiSBVnOD44kLDwJl9nH8Q
    """
    g = 9.81
    def __init__(self, mass, distance_cg_motor, J, lift_factor, drag_factor):
        """ 
        Arguments:
            mass: mass of the quadrotor.
            distance_cg_motor: distance between the center 
            of gravity of the quadrotor and a motor.
            J: inertia matrix.
            lift factor: lift factor.
            drag_factor: drag factor.
        Returns:
            None
        """
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
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
        )

        self.C_faulty = np.array(self.C)

        self.l = distance_cg_motor
        self.b = lift_factor
        self.d = drag_factor

    def get_state_space(self):
        """ This function returns the states of the multi-
            agent system.
        """
        return self.A, self.B, self.C_faulty

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

    def add_fault_to_agent(self, type = "random"):
        if type == "x":
            self.C_faulty[0, 0] = 0
        elif type == "y":
            self.C_faulty[1, 1] = 0
        elif type == "z":
            self.C_faulty[2, 2] = 0
        elif type == "phi":
            self.C_faulty[3, 3] = 0
        elif type == "theta":
            self.C_faulty[4, 4] = 0
        elif type == "psy":
            self.C_faulty[5, 5] = 0
        else:
            random = np.random.randint(0, 5)
            self.C_faulty[random, random] = 0

        return self.C_faulty