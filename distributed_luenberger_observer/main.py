from classes.graph import *
from control import place
from classes.parameters_function import *
from observer import *

np.random.seed(1)

nbr_agent = 4

step = 0.01
t_max = 50
nbr_step = int(t_max/step)

Adj = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
G = Graph(4, Adj)
Degree = G.access_Degree()
Laplacien = G.access_Laplacien()
print(Laplacien)

# A_sys = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
# A_sys = np.array([[2, -1.5, 0, 0], [0, 1 , 0, 0], [0, 1, -1.5, 0], [0, 0, 0, -2.3]])
# A_sys = np.array([[-2, -1, 0, 0], [0, -1 , 0, 0], [0, -1, -1.5, 0], [0, 0, 0, -2.3]])

size_system = 4

A_sys = random_std_form_matrix(size_system)
print("not noisy", A_sys)

noise = False

std_noise = np.abs(np.mean(A_sys[np.where(A_sys != 0)]))/10
if noise:
    A_sys_noisy = noisy_system(A_sys, std_noise, "other")
    print("std", std_noise, "mean", np.mean(A_sys[np.where(A_sys != 0)]))
    print("noisy", A_sys_noisy)
else:
    A_sys_noisy = A_sys

print("eig system", np.linalg.eig(A_sys)[0])

C = np.eye(4)
C_sys = diag((C[:,0].reshape((1,-1)), C[:,1].reshape((1,-1)), C[:,2].reshape((1,-1)), C[:,3].reshape((1,-1))))
B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])


eig = -1*np.abs(np.random.rand(np.shape(A_sys)[0])+0.5)
# eig = [-10, -15, -7.5, -8]
print("eig of commanded sys", eig)
K_sys = place(A_sys_noisy, B_sys, eig)

# K_sys = np.zeros((np.shape(B_sys)[1], np.shape(A_sys)[0]))

# k = [3.5, 4.5, 1.7, 3]

gamma = 6

M, L, T, Md, Ld, M_dict, L_dict = parameters(A_sys, B_sys, C, nbr_agent)

# check_parameters(k, gamma, A_sys, nbr_agent, Laplacien, T, size_obsv, 1.1)

size_A = np.shape(A_sys)[0]
x, x_hat, x_hat_concatenated = initialize_state_values(size_A, nbr_step, nbr_agent)

size_C = np.shape(C_sys)[0]

y, y_hat, y_concatenated, S_y_concatenated = initialize_output_values(nbr_step, nbr_agent, size_A, size_C)

A_sys_concatenated, A_sys_noisy_concatenated, B_sys_concatenated, K_sys_concatenated = initialize_concatenated(nbr_agent, A_sys, A_sys_noisy, B_sys, K_sys)

size_B_concatenated = np.shape(B_sys_concatenated)[1]
size_B = 2
u_sys, u_concatenated = input(step, t_max, nbr_agent, size_B_concatenated, size_B)

Laplacien_m = np.kron(Laplacien, np.eye(np.shape(A_sys)[0]))

k_adapt = np.ones((nbr_agent, nbr_step))

Se2 = np.zeros((nbr_agent, np.shape(A_sys)[0], nbr_step))

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(A_sys, x[ :,i]) - step*np.dot(np.dot(B_sys, K_sys), x_hat[0,:,i]) + step*np.reshape(np.dot(B_sys, u_sys[:,i]), (-1,)) + x[:,i] 
    x_concatenated = np.array([x[:, i] for _ in range(nbr_agent)])
    x_concatenated = np.reshape(x_concatenated, (np.shape(x_concatenated)[0]*np.shape(x_concatenated)[1], ))

    
    # y_concatenated[:,i+1] = np.dot(C_sys, x[:,i+1])
    y_concatenated[:, i+1] = np.dot(C_sys, x_concatenated)
    S_y_concatenated += step*y_concatenated[:, i+1]
    # y_concatenated[:,i+1] = np.reshape(y_hat[:,:,i+1] , (nbr_agent*nbr_agent,))

    x_hat_concatenated[:,i+1] = step*np.dot(A_sys_noisy_concatenated - np.dot(B_sys_concatenated, K_sys_concatenated), x_hat_concatenated[:,i]) + step*np.reshape(np.dot(B_sys_concatenated, u_concatenated[:,i]), (-1, )) + x_hat_concatenated[:,i]  + step*np.dot(np.dot(L, C_sys), x_concatenated - x_hat_concatenated[:,i]) + step*gamma*np.dot(np.dot(np.linalg.inv(M), -Laplacien_m), x_hat_concatenated[:, i])
    x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (nbr_agent, np.shape(A_sys)[0],))

    for ind in range(nbr_agent):
        k_adapt[ind,i+1] = k_adapt[ind,i] + step*np.linalg.norm(np.dot(Laplacien[ind,:],(x_hat[:,:, i+1] - x_hat[ind,:, i+1])**2), 2)
        size_obsv = np.linalg.matrix_rank(obsv(A_sys, C[:, ind]))
        M_dict[str(ind)] = Mi(T[str(ind)], k_adapt[ind, i+1], Md[str(ind)], np.shape(A_sys)[0] - size_obsv)
        L_dict[str(ind)] = Li(T[str(ind)], np.reshape(Ld[str(ind)], (-1,)).T,  np.shape(A_sys)[0] - size_obsv)

    M = [v for v in M_dict.values()]
    M = diag(M)

    L = [v for v in L_dict.values()]
    L = diag(L)


    Se2[:,:,i+1] = step*(x_hat[:,:, i+1] - x[:,i+1])**2 + Se2[:,:,i]

print("gain k after adaptive algo", k_adapt[:, nbr_step-1])
print(np.sum(Se2, axis=0)[:, nbr_step -1])

plot_states(x, x_hat, t_max, step)
plot_k(nbr_agent, t_max, step, k_adapt)

plt.plot(np.arange(0,t_max, step), np.transpose(np.sum(Se2, axis=0)))
plt.grid()
plt.show()