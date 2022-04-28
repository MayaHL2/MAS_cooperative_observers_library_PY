import matplotlib.pyplot as plt
import numpy as np
from .classes.graph import *
from .classes.parameters_function import *
from control import place

nbr_agent = 4

step = 0.01
t_max = 20
nbr_step = int(t_max/step)

Adj = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
Laplacien = -1*np.array([[-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]])

A_sys = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
C_sys = np.eye(4)
C_sys = diag((C_sys[:,0].reshape((1,-1)), C_sys[:,1].reshape((1,-1)), C_sys[:,2].reshape((1,-1)), C_sys[:,3].reshape((1,-1))))
# B_sys = np.reshape(np.eye(np.shape(A_sys)[0])[np.shape(A_sys)[0] -1], (np.shape(A_sys)[0], 1))
B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])

eig = -np.abs(np.random.rand(np.shape(A_sys)[0])+0.75)
K_sys = place(A_sys, B_sys, eig)

T1 = T2 = np.eye(4)
T3 = T4 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

L1d = np.transpose([3, 1])
L2d = np.transpose([-1, 3])
L3d = np.transpose([7, 4])
L4d = np.transpose([-4, 7])

M1d = np.array([[0.5, -0.5], [0.5, 1]])
M2d = np.array([[1, 0.5], [0.5, 0.5]])
M3d = np.array([[0.286, -0.25], [-0.25, 0.387]])
M4d = np.array([[0.387, 0.25], [0.25, 0.286]])

gamma = 6
k = {"1": 3, "2": 4.5, "3":4, "4": 4.5}
k1 = 3
k2 = 4.5
k3 = 4 
k4 = 4.5

# k = np.transpose([k1, k2, k3, k4])

M1 = Mi(T1, k["1"], M1d, 2)
M2 = Mi(T2, k["2"], M2d, 2)
M3 = Mi(T3, k["3"], M3d, 2)
M4 = Mi(T4, k["4"], M4d, 2)
M = diag((M1, M2, M3, M4))

L1 = Li(T1, L1d, 2)
L2 = Li(T2, L2d, 2)
L3 = Li(T3, L3d, 2)
L4 = Li(T4, L4d, 2)

L = diag((np.reshape(L1, (np.shape(L1)[0], 1)), np.reshape(L2, (np.shape(L2)[0], 1)), np.reshape(L3, (np.shape(L3)[0], 1)), np.reshape(L4, (np.shape(L4)[0], 1))))

x = np.zeros((np.shape(A_sys)[0], nbr_step))
x[:, 0] = np.transpose([1, 0.5, 1, 0])
x_hat = 5*np.random.rand(nbr_agent, np.shape(A_sys)[0], nbr_step)
x_hat_concatenated = np.reshape(x_hat, (nbr_agent*np.shape(A_sys)[0], nbr_step))


y = np.zeros((np.shape(C_sys)[0], nbr_step))
y_hat = np.zeros((nbr_agent, np.shape(C_sys)[0], nbr_step))

A_sys_concatenated = np.kron(np.eye(nbr_agent), A_sys)
# A_sys_concatenated_noisy = np.random.normal(np.mean(A_sys_concatenated), 0.01, np.shape(A_sys_concatenated))
B_sys_concatenated = np.kron(np.eye(nbr_agent), B_sys)
K_sys_concatenated = np.kron(np.eye(nbr_agent), K_sys)

# u_sys = np.cos(np.arange(0, t_max, step))
u_sys = np.ones((2, nbr_step))
u_concatenated = np.reshape(np.array([u_sys for _ in range(nbr_agent)]), (np.shape(B_sys_concatenated)[1], -1))


Laplacien_m = np.kron(Laplacien, np.eye(np.shape(A_sys)[0]))

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(A_sys - np.dot(B_sys, K_sys),x[ :,i]) + step*np.reshape(np.dot(B_sys, u_sys[:,i]), (-1,)) + x[:,i] 
    x_concatenated = np.array([x[:, i] for _ in range(nbr_agent)])
    x_concatenated = np.reshape(x_concatenated, (np.shape(x_concatenated)[0]*np.shape(x_concatenated)[0], ))

    
    # y[:,i+1] = np.dot(C_sys, x[:,i+1])
    # y_concatenated[:,i+1] = np.reshape(y_hat[:,:,i+1] , (nbr_agent*nbr_agent,))

    x_hat_concatenated[:,i+1] = step*np.dot(A_sys_concatenated - np.dot(B_sys_concatenated, K_sys_concatenated), x_hat_concatenated[:,i]) + step*np.reshape(np.dot(B_sys_concatenated, u_concatenated[:,i]), (-1, )) + x_hat_concatenated[:,i]  + step*np.dot(np.dot(L, C_sys), x_concatenated - x_hat_concatenated[:,i]) + step*gamma*np.dot(np.dot(np.linalg.inv(M), -Laplacien_m), x_hat_concatenated[:, i])
    x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (nbr_agent, np.shape(A_sys)[0],))

    # M1 = Mi(T1, k["1"], M1d, 2)
    # M2 = Mi(T2, k["2"], M2d, 2)
    # M3 = Mi(T3, k["3"], M3d, 2)
    # M4 = Mi(T4, k["4"], M4d, 2)
    # M = diag((M1, M2, M3, M4))

for j in range(x.shape[0]):
    # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]))
    # plt.plot(np.arange(0,t_max, step), np.transpose(x_hat[:, j,:]))
    plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 
    plt.grid()
    plt.show()