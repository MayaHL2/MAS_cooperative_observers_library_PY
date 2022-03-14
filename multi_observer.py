import numpy as np
import matplotlib.pyplot as plt
from graph import *

nbr_agent = 3
size_agent = 4

step = 0.01
t_max = 10
nbr_step = int(t_max/step)

Adj = random_Adjacency(nbr_agent)
D = Degree_matrix(Adj)
L = Laplacien_matrix(Adj, D)

x = 5*np.random.rand(size_agent*nbr_agent, nbr_step)
x_hat = 5*np.random.rand(nbr_agent, size_agent*nbr_agent, nbr_step)
x_concatenated = x_hat_concatenated = np.reshape(x_hat, (nbr_agent*size_agent*nbr_agent, nbr_step))

A = -10*np.abs(np.diag(np.random.rand(size_agent, ))) # Stable
B = np.reshape(np.identity(size_agent)[size_agent-1], (size_agent, 1))
C = np.reshape(np.identity(size_agent)[0], (1, size_agent))

y = np.zeros((nbr_agent*np.shape(C)[0], nbr_step))
y_hat = np.zeros((nbr_agent, nbr_agent*np.shape(C)[0], nbr_step))
y_concatenated = np.zeros((nbr_agent*nbr_agent*np.shape(C)[0], nbr_step))

# observability = np.linalg.matrix_rank(co.obsv(A, C)) == size_agent

# print(np.linalg.eigvals(A))
# print(np.linalg.matrix_rank(co.obsv(A, C)), co.obsv(A, C))


I_n = np.eye(size_agent)
I_m = np.eye(nbr_agent)
I_mn = np.eye(nbr_agent*size_agent)
L_mn = np.kron(I_mn, L)
A_sys = np.kron(L,  A)
B_sys = np.kron(I_m, B)
C_sys = np.kron(I_m, C)

A_sys_concatenated = np.kron(I_m, A_sys)
B_sys_concatenated = np.kron(I_m, B_sys)
C_sys_concatenated = np.kron(I_m, C_sys)

u = np.ones((np.shape(B)[1], nbr_step))

u_sys = np.array([u for _ in range(nbr_agent)])
u_concatenated = np.array([u_sys for _ in range(nbr_agent)])
u_concatenated = np.reshape(u_concatenated, (nbr_agent*nbr_agent, np.shape(B)[1], nbr_step))

K1 = step*np.ones((np.shape(A_sys_concatenated)[0],np.shape(C_sys_concatenated)[0]))
K2 = step*np.ones((nbr_agent*nbr_agent*size_agent,np.shape(L_mn)[0]))

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(A_sys, x[:,i]) + np.reshape(np.dot(B_sys, u_sys[:,:,i]), (12,)) + x[:,i]
    x_concatenated[:, i+1] = np.reshape(np.array([x[:,i+1] for _ in range(nbr_agent)]),  (nbr_agent*size_agent*nbr_agent, )) # Il ne sert à rien

    y[:,i+1] = np.dot(C_sys, x[:,i+1])
    y_concatenated[:,i+1] = np.reshape(y_hat[:,:,i+1] , (nbr_agent*nbr_agent, ))

    x_hat_concatenated[:,i+1] = step*np.dot(A_sys_concatenated+np.dot(K1,C_sys_concatenated), x_hat_concatenated[:,i]) + np.reshape(np.dot(B_sys_concatenated, u_concatenated[:,:,i]), (-1, )) + np.dot(K2,np.dot(L_mn, x_hat_concatenated[:,i])) + x_hat_concatenated[:,i] 
    x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (nbr_agent, size_agent*nbr_agent,))



for j in range(np.shape(x_hat)[1]):
    # plt.plot(np.arange(0,t_max, step), np.transpose(x_hat[:,j,:]))
    # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]))
    plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 
    plt.grid()
    plt.show()



# for j in range(size_agent*nbr_agent):
#     plt.plot(np.arange(0,t_max, step), x[j,:])

#     plt.grid()
#     plt.show()