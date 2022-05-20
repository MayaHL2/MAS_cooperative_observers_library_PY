from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
from K import K

np.random.seed(3)

G = Graph(2, [[0, 1], [1, 0]])

print(G.Adj)

# A_sys = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-2.7, -3.5, -1.4, -2.5]])
# A_sys = np.array([[-2, -1, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, -1, -1.5, 0],
#                   [0, 0, 0, -2.3]])
# print(np.linalg.eig(A_sys))

# t_response = list()

std_percent = 0

# std = std_percent*np.abs(np.mean(A_sys))

# B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])
# C = np.eye(4)
# C_sys = (np.array([[1, 0, 1, 1],[1, 0, 4, 7]]),np.array([[3, 1, 1, 1], [0, 0, -4, 0]]), np.array([[1, 0, 1, 0], [0, 1, 1, 1]]), np.array([[1, 6, 0, 2],  [0, 0, 5, 1]]))
# C_sys = (np.reshape(C[:,0], (1, -1)), np.reshape([0, 1, 0, 0], (1, -1)), np.reshape([0, 0, 1, 0], (1,-1)), np.reshape(C[:,3], (1, -1)))

J = [0.0820, 0.0845, 0.1377]
m = 4.34
d = 0.315

lift_factor = 1.5*10**(-9)
drag_factor = 6.11*10**((-8))
UAV = Quadrotor(m, d, J, lift_factor, drag_factor)
A_sys, B_sys, C_sys = UAV.get_state_space()

C_sys = (C_sys, C_sys)
x = np.zeros(np.shape(A_sys)[0])

MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)

print(MA.is_jointly_obsv())
print(MA.obsv_index())

eig = MA.feedback_control(feedback_gain = K)
# eig = MA.feedback_control([-4.7540,   -5.5407,   -3.4545,   -4.5541,   -6.4647,   -6.4815,   -2.6279,   -3.0362,   -2.2573,   -4.2040,   -2.1494, -4.2842,   -5.2457,   -3.3924,   -5.3813,   -4.9543,   -2.1199,   -4.7943,   -3.2963,   -4.0755,   -3.4176,   -5.4657, -4.2023,   -2.7843])

MA.step_response(7, stabilized= True)