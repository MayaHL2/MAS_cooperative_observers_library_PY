import numpy as np
import control as co

def initialize_random(size_agent, nbr_agent, nbr_step, range_initial_x, range_initial_x_hat, type_A = "standard"):
    x = range_initial_x*np.random.rand(size_agent*nbr_agent, nbr_step)
    x_hat = range_initial_x_hat*np.random.rand(nbr_agent, size_agent*nbr_agent, nbr_step)
    x_concatenated = x_hat_concatenated = np.reshape(x_hat, (nbr_agent*size_agent*nbr_agent, nbr_step))

 
    if type_A == "standard":
        A = np.zeros((size_agent,size_agent))
        A[0:size_agent-1,1:] = np.eye(size_agent-1)
        A[size_agent-1:size_agent,0:] = -np.abs(np.random.rand(1, size_agent))
    elif type_A == "diag":
        A = -10*np.abs(np.diag(np.random.rand(size_agent, ))) # Stable matrix in

    # print("eig A", np.linalg.eigvals(A))


    B = np.reshape(np.identity(size_agent)[size_agent-1], (size_agent, 1))
    C = np.reshape(np.identity(size_agent)[0], (1, size_agent))

    y = np.zeros((nbr_agent*np.shape(C)[0], nbr_step))
    y_hat = np.zeros((nbr_agent, nbr_agent*np.shape(C)[0], nbr_step))
    y_concatenated = np.zeros((nbr_agent*nbr_agent*np.shape(C)[0], nbr_step))

    return x, x_hat, x_hat_concatenated, x_concatenated, A, B, C, y, y_hat, y_concatenated

def define_multi_agent_system(size_agent, nbr_agent, L, A, B, C, K):   
    L_mn = np.kron(np.eye(nbr_agent), np.kron(L, np.eye(size_agent)))
    A_sys = np.kron(np.eye(nbr_agent),  A)
    B_sys = np.kron(np.eye(nbr_agent), B)
    C_sys = np.kron(np.eye(nbr_agent), C)
    K_sys = np.kron(np.eye(nbr_agent), K)

    
    # print("eig A_sys", np.linalg.eigvals(A))
    # print("eig A_sys - B_sys*K_sys", np.linalg.eigvals(A_sys - np.dot(B_sys, K_sys)))
    # print(K)

    return L_mn, A_sys, B_sys, C_sys, K_sys

def concatenate_sys(nbr_agent, A_sys, B_sys, C_sys, K_sys):
    A_sys_concatenated = np.kron(np.eye(nbr_agent), A_sys)
    B_sys_concatenated = np.kron(np.eye(nbr_agent), B_sys)
    C_sys_concatenated = np.kron(np.eye(nbr_agent), C_sys)
    K_sys_concatenated = np.kron(np.eye(nbr_agent), K_sys)

    return A_sys_concatenated, B_sys_concatenated, C_sys_concatenated, K_sys_concatenated

def input(nbr_step, nbr_agent, shape_u, type = "unit"):
    u = np.ones((shape_u, nbr_step))
    u_sys = np.array([u for _ in range(nbr_agent)])
    u_concatenated = np.array([u_sys for _ in range(nbr_agent)])
    u_concatenated = np.reshape(u_concatenated, (nbr_agent*nbr_agent, shape_u, nbr_step))

    return u, u_sys, u_concatenated

def is_observable(A, C, size_agent):
    observability = np.linalg.matrix_rank(co.obsv(A, C)) == size_agent
    return observability