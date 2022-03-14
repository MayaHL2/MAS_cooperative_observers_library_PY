import control as co
import matplotlib.pyplot as plt
from graph import *

nbr_agent = 4

Adj = random_Adjacency(nbr_agent)
D = Degree_matrix(Adj)
L = Laplacien_matrix(Adj, D)

L = [ [1.5121185,  -0.62930889, -0.18984814, -0.69296147],
 [-0.42682207,  1.05859722, -0.16422574, -0.46754941],
 [-0.2041849,  -0.15189416,  0.50364008, -0.14756101],
 [-0.33584794, -0.35120307, -0.45345045,  1.14050146]]

# Adj = 1 - np.eye(2) 
# D = Degree_matrix(Adj)
# L = Laplacien_matrix(Adj, D)

# print(L)

# A = 0
# B = 1
# C = 1
A = [[1, 0], [0, 1]]
B = [[0],[1]]
C = [1, 0]

sys_agent = co.StateSpace(A, B, C, 0)

# t1, y1 = co.step_response(sys_agent)
# plt.grid()
# plt.plot(t1,y1)
# plt.show()

# B = 1
# C = 1

n = np.shape(A)[0]

I_n = np.eye(n)
I_m = np.eye(nbr_agent)
A_sys = np.kron(L,  A)
B_sys = np.kron(I_m, B)
C_sys = np.kron(I_m, I_n)

# print(np.shape(A_sys), np.shape(B_sys), np.shape(C_sys))

# C_sys = np.kron(I_m, C)

A_sys_controlled = np.kron(L,A) - np.kron(L, I_n) 

sys = co.StateSpace(A_sys_controlled, B_sys, C_sys, 0)

print("eig", np.linalg.eigvals(A_sys_controlled))

t, y = co.step_response(sys)
y = np.transpose(np.sum(y, axis = 1))
print(np.shape(y), np.shape(t))

for col in range(np.shape(y)[1]):
    plt.plot(t, y[:, col])
    # plt.plot(y)

    plt.grid()
    plt.show()