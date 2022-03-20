import matplotlib.pyplot as plt
from graph import *
from constant import *
from multi_observer_function import *

Adj = random_Adjacency(nbr_agent)
D = Degree_matrix(Adj)
L = Laplacien_matrix(Adj, D)

x, x_hat, x_hat_concatenated, x_concatenated, A, B, C, y, y_hat, y_concatenated = initialize_random(size_agent, nbr_agent, nbr_step, range_initial_x, range_initial_x_hat)

K = co.place(A, B, -1.5*np.abs(np.random.rand(size_agent)+0.75))
# print("eig(A-BK)", np.linalg.eigvals(A - np.dot(B,K)))

L_mn, A_sys, B_sys, C_sys, K_sys = define_multi_agent_system(size_agent, nbr_agent, L, A, B, C, K)
A_sys_concatenated, B_sys_concatenated, C_sys_concatenated, K_sys_concatenated = concatenate_sys(nbr_agent, A_sys, B_sys, C_sys, K_sys)

shape_u = np.shape(B)[1]
u, u_sys, u_concatenated = input(nbr_step, nbr_agent, shape_u)

K1 = co.place(np.transpose(A_sys), np.transpose(C_sys), -np.abs(np.random.rand(np.shape(A_sys)[0])))
K1 = np.transpose(K1)

# print("eig(A+LC)", np.linalg.eigvals(A_sys- np.dot(K1,C_sys)))
K1 = 0*step*np.kron(np.eye(nbr_agent), K1)
# print(np.shape(K1))
# print("eig(A+LC)1", np.linalg.eigvals(A_sys_concatenated - np.dot(K1,C_sys_concatenated)))

# K1 = step**4*np.ones((np.shape(A_sys_concatenated)[0],np.shape(C_sys_concatenated)[0]))
K2 = 0*step*np.ones((nbr_agent*nbr_agent*size_agent,np.shape(L_mn)[0]))

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(A_sys - np.dot(B_sys, K_sys), x_hat[0, :,i]) + np.reshape(np.dot(B_sys, u_sys[:,:,i]), (-1,)) + x[:,i]

    y[:,i+1] = np.dot(C_sys, x[:,i+1])
    y_concatenated[:,i+1] = np.reshape(y_hat[:,:,i+1] , (nbr_agent*nbr_agent,))

    x_hat_concatenated[:,i+1] = step*np.dot(A_sys_concatenated - np.dot(B_sys_concatenated, K_sys_concatenated)+np.dot(K1,C_sys_concatenated), x_hat_concatenated[:,i]) + np.reshape(np.dot(B_sys_concatenated, u_concatenated[:,:,i]), (-1, )) + np.dot(K2,np.dot(L_mn, x_hat_concatenated[:,i])) + x_hat_concatenated[:,i] 
    x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (nbr_agent, size_agent*nbr_agent,))

# print("eig(A+LC)", np.linalg.eigvals(A_sys+ step*np.dot(np.ones((np.shape(A_sys)[0], np.shape(C_sys)[0])),C_sys)))


for j in range(np.shape(x_hat)[1]//nbr_agent):
    # plt.plot(np.arange(0,t_max, step), np.transpose(u[0])) 
    plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]), "b")
    # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 
    plt.plot(np.arange(0,t_max, step), np.transpose(x_hat[:,j,:]), "g")
    # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:] - u[0]), "b")

    plt.grid()
    plt.show()