import numpy as np
from control import obsv
from harold import staircase
from classes.quadrotor import *
from classes.helper_function import *

# A = np.array([[-2, 0, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, 0, -1.5, -5],
#                   [0, 0, 0, -2.3]])


# B = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])

# C = np.array([[1, 0, 0, 0]])

# # J = [0.0820, 0.0845, 0.1377]
# # m = 4.34
# # d = 0.315

# # lift_factor = 1.5*10**(-9)
# # drag_factor = 6.11*10**((-8))
# # UAV = Quadrotor(m, d, J, lift_factor, drag_factor)
# # A, B, C = UAV.get_state_space()

# nbr_iteration = 5
# obsv_ind = np.linalg.matrix_rank(obsv(A, C))

# it = 0
# added_obs_states = C
# while obsv_ind!= A.shape[0] and it < nbr_iteration:

#     _, _, _, T = staircase(A, B, C, form = "o")
#     obsv_ind = np.linalg.matrix_rank(obsv(A, C))

#     print(obsv_ind)
#     P = np.zeros(A.shape)
#     P[obsv_ind:, :] = T[obsv_ind:, :]
#     print(T)
#     print(P)

#     added_obs_states = added_obs_states + np.sum(P, axis = 0)
#     obsv_ind = np.linalg.matrix_rank(obsv(A, added_obs_states))

#     obsv_ind = A.shape[0]

#     it += 1

# print(obsv_ind)
# added_obs_states = np.float16(np.logical_or(added_obs_states> 10**(-6), added_obs_states<- 10**(-6))) - C
# C = np.row_stack((C, added_obs_states))
# print(C)

# print(np.linalg.matrix_rank(obsv(A, C)))


# print(gaussian_noise([0, 100], [0.1, 2], (2,)))

nbr_agents = 6
faulty_agent = np.array([1, 3, 4])
A = np.zeros(nbr_agents + len(faulty_agent)**2)
# x = list([])

# for i in faulty_agent:
#     A[i] = [0, 1, 1]

# B = np.zeros(nbr_agents)
# j = 0
# for u in A:
#     if not isinstance(u, int):
#         print(u)
#     else:
#         print(u, "is int")
#     j += 1

# print(B)

# for i in faulty_agent:
#     x = np.ones(len(faulty_agent) + 1)
#     x[0] = 0
#     A[np.arange(i, i + len(faulty_agent) + 1)] = x

#     faulty_agent = len(faulty_agent) + faulty_agent

#     print(i, faulty_agent)

# print(A)

added_lines = np.zeros(len(faulty_agent)**2)

k = 0
for i in faulty_agent:
    print(added_lines[k*(len(faulty_agent)):(k +1)*len(faulty_agent)])
    added_lines[k*(len(faulty_agent)):(k +1)*len(faulty_agent)] = np.arange(i + k*len(faulty_agent) + 1, i + (k+1)*len(faulty_agent) + 1)
    k+=1

A[np.int8(added_lines)] = 1

print(A)