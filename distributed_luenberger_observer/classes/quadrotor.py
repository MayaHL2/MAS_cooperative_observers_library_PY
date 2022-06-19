import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

    def draw_quadrotor(self, p, angles, t_max, step_UAV, distance_wings, color_path = "green", color_quadrotor = "#000000"):
        
        # fig = plt.figure()
        
        # syntax for 3-D projection
        ax = plt.axes(projection ='3d')
        # ax.set(xlim=(-np.max(p[0:3, :]), np.max(p[0:3, :])), ylim=(-np.max(p[0:3, :]), np.max(p[0:3, :])), zlim=(-np.max(p[0:3, :]), np.max(p[0:3, :])))
           
        # plotting
        ax.plot3D(p[0, :], p[1, :], p[2,:], color_path)

        p_cg = p[:, np.arange(0, int(t_max/0.01), step_UAV)]
        angles_drone = angles[:, np.arange(0, int(t_max/0.01), step_UAV)]

        # distance_wings = np.max(p[0:3, :])/7

        p1 = list([p_cg[:, 0] + [distance_wings, 0, 0]])
        p2 = list([p_cg[:, 0] + [-distance_wings, 0, 0]])
        p3 = list([p_cg[:, 0] + [0, distance_wings, 0]])
        p4 = list([p_cg[:, 0] + [0, -distance_wings, 0]])

        for i in range(1, int(t_max/(0.01*step_UAV))):
            r = R.from_euler('xyz', angles_drone[:, i]*180/np.pi, degrees = True)
            r = r.as_matrix()

            p1.append(np.dot(r,[0, distance_wings, 0] )+ p_cg[:, i])
            p2.append(np.dot(r,[0, -distance_wings,0])+ p_cg[:, i])
            p3.append(np.dot(r,[0, 0, distance_wings] )+ p_cg[:, i])
            p4.append(np.dot(r,[0, 0, -distance_wings])+ p_cg[:, i])

            ax.scatter(np.array(p_cg)[0,i], np.array(p_cg)[1,i], np.array(p_cg)[2,i], color = color_quadrotor)

            # v = np.array(p1)[i,:] - p_cg[:, i]
            # ax.plot([p_cg[0, i],  p_cg[0, i] + v[0]], [p_cg[1, i], p_cg[1, i] + v[1]], [p_cg[2, i], p_cg[2, i] + v[2]], color = color_quadrotor, linewidth = 5)
            # ax.plot([p_cg[0, i],  p_cg[0, i] - v[0]], [p_cg[1, i], p_cg[1, i] - v[1]], [p_cg[2, i], p_cg[2, i] - v[2]], color = color_quadrotor, linewidth = 5)
            # r = R.from_euler('x', 90, degrees = True)
            # r = r.as_matrix()
            # v = np.dot(r, v)
            # ax.plot([p_cg[0, i],  p_cg[0, i] + v[0]], [p_cg[1, i], p_cg[1, i] + v[1]], [p_cg[2, i], p_cg[2, i] + v[2]], color = color_quadrotor, linewidth = 5)
            # ax.plot([p_cg[0, i],  p_cg[0, i] - v[0]], [p_cg[1, i], p_cg[1, i] - v[1]], [p_cg[2, i], p_cg[2, i] - v[2]], color = color_quadrotor, linewidth = 5)


            ax.plot([p_cg[0, i],  np.array(p1)[i,0]], [p_cg[1, i], np.array(p1)[i,1]], [p_cg[2, i], np.array(p1)[i,2]], color = color_quadrotor, linewidth = 5)
            ax.plot([p_cg[0, i],  np.array(p2)[i,0]], [p_cg[1, i], np.array(p2)[i,1]], [p_cg[2, i], np.array(p2)[i,2]], color = color_quadrotor, linewidth = 5)
            ax.plot([p_cg[0, i],  np.array(p3)[i,0]], [p_cg[1, i], np.array(p3)[i,1]], [p_cg[2, i], np.array(p3)[i,2]], color = color_quadrotor, linewidth = 5)
            ax.plot([p_cg[0, i],  np.array(p4)[i,0]], [p_cg[1, i], np.array(p4)[i,1]], [p_cg[2, i], np.array(p4)[i,2]], color = color_quadrotor, linewidth = 5)

        # t = np.arange(0, t_max, 0.01)

        # x = np.array([0.4*t, 0.4*np.sin(np.pi*t), 0.6*np.cos(np.pi*t)])
        # ax.plot3D(x[0, :], x[1, :], x[2, :], color_path)

        # p_cg = x[:, np.arange(0, int(t_max/0.01), step_UAV)]

        # p0 = np.array([np.cos(0.2*np.pi*t), np.sin(0.2*np.pi*t), np.zeros(np.max(np.shape(t),))]) + x
        # r = R.from_euler('x', 90, degrees = True).as_matrix()


        # for i in range(1, int(t_max/(0.01*step_UAV))):
        #     p1 = np.dot(r, p0[:, i])
        #     # p1 = p0[:, i]
        #     # r1 = R.from_euler('x', 90, degrees = True).as_matrix()
        #     p2 = p0[:, i]

        #     ax.plot([p_cg[0, i], p_cg[0, i] + np.array(p1)[0]], [p_cg[1, i], p_cg[1, i] + np.array(p1)[1]], [p_cg[2, i], p_cg[2, i] + np.array(p1)[2]], color = color_quadrotor, linewidth = 5)
        #     ax.plot([p_cg[0, i], p_cg[0, i] + np.array(p2)[0]], [p_cg[1, i], p_cg[1, i] + np.array(p2)[1]], [p_cg[2, i], p_cg[2, i] + np.array(p2)[2]], color = color_quadrotor, linewidth = 5)

        plt.show()
