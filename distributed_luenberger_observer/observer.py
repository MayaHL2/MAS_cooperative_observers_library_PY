from parameters_function import *
from harold import staircase
import matplotlib.pyplot as plt
from control import obsv


def random_std_form_matrix(size_system):
    A_sys = np.zeros((size_system,size_system))
    A_sys[0:size_system-1,1:] = np.eye(size_system-1)
    A_sys[size_system-1:size_system,0:] = -5*np.abs(np.random.rand(1, size_system))
    return A_sys

def noisy_system(A_sys, std_noise, form = "standard"):
    if form == "standard":
        A_sys_noisy = np.array(A_sys)
        A_sys_noisy[-1] = np.random.normal(np.mean(A_sys[-1]), std_noise, np.shape(A_sys[-1]))
    else:
        A_sys_noisy = np.array(A_sys)
        A_sys_noisy += np.random.normal(0, std_noise, np.shape(A_sys))
        A_sys_noisy[np.where(A_sys == 0)] = 0

    return A_sys_noisy

def parameters(A_sys, B_sys, C, nbr_agent):
    T = dict()
    A_bar = dict()
    B_bar = dict()
    C_bar = dict()

    Ad = dict()
    Hd = dict()
    Md = dict()
    Ld = dict()
    L_dict = dict()
    M_dict = dict()

    k = np.ones(nbr_agent)

    for i in range(nbr_agent):
        size_obsv = np.linalg.matrix_rank(obsv(A_sys, C[:, i]))
        print("size obsv of agent", i+1, "is" , size_obsv)
        # Should I put O.T or O in the transformation matrix
        # T[str(i)] = transformation_matrix(O.T, size_obsv, nbr_agent)
        A_bar[str(i)], B_bar[str(i)], C_bar[str(i)], T[str(i)] = staircase(A_sys, B_sys, np.reshape(C[:,i], (1,np.shape(A_sys)[0])), form = "o")
        # A_bar[str(i)], B_bar[str(i)], C_bar[str(i)] = new_basis(A_sys, B_sys, C[:,i], T[str(i)])
        Ad[str(i)], _, _ = separate_A_bar(A_bar[str(i)], size_obsv)
        Hd[str(i)] = Hid(C_bar[str(i)], size_obsv)
        Ld[str(i)] = Lid(Ad[str(i)], Hd[str(i)])
        Md[str(i)] = Mid(Ad[str(i)], Ld[str(i)], Hd[str(i)])
        M_dict[str(i)] = Mi(T[str(i)], k[i], Md[str(i)], np.shape(A_sys)[0] - size_obsv)
        L_dict[str(i)] = Li(T[str(i)], np.reshape(Ld[str(i)], (-1,)).T,  np.shape(A_sys)[0] - size_obsv)

        # print("numéro", i)
        # print("T", T[str(i)])
        # print("L", Ld[str(i)])
        # print("M", Md[str(i)])
        # print()

        # print("numéro", i)
        # print("T", T[str(i)])
        # print("A", A_bar[str(i)])
        # print("B", B_bar[str(i)])
        # print("C", C_bar[str(i)])
        # print()


    M = [v for v in M_dict.values()]
    M = diag(M)

    L = [v for v in L_dict.values()]
    L = diag(L)

    return M, L, T, Md, Ld, M_dict, L_dict


def initialize_state_values(size_A, nbr_step, nbr_agent, initial_values_x = [1, 0.5, 1, 0]):
    x = np.zeros((size_A, nbr_step))
    x[:, 0] = np.transpose(initial_values_x)
    x_hat = 7*np.random.rand(nbr_agent, size_A, nbr_step)
    print("inital values of x_hat of agent 1", x_hat[0,:,0])
    print("inital values of x_hat of agent 2", x_hat[1,:,0])
    print("inital values of x_hat of agent 3", x_hat[2,:,0])
    print("inital values of x_hat of agent 4", x_hat[3,:,0])

    x_hat_concatenated = np.reshape(x_hat, (nbr_agent*size_A, nbr_step))

    return x, x_hat, x_hat_concatenated

def initialize_output_values(nbr_step, nbr_agent, size_A, size_C):
    y = np.zeros((size_C, nbr_step))
    y_hat = np.zeros((nbr_agent, size_C, nbr_step))
    y_concatenated = np.zeros((size_A, nbr_step))
    S_y_concatenated = 0
    return y, y_hat, y_concatenated, S_y_concatenated

def initialize_concatenated(nbr_agent, A_sys, A_sys_noisy, B_sys, K_sys):
    A_sys_concatenated = np.kron(np.eye(nbr_agent), A_sys)
    A_sys_noisy_concatenated = np.kron(np.eye(nbr_agent), A_sys_noisy)
    B_sys_concatenated = np.kron(np.eye(nbr_agent), B_sys)
    K_sys_concatenated = np.kron(np.eye(nbr_agent), K_sys)

    return A_sys_concatenated, A_sys_noisy_concatenated, B_sys_concatenated, K_sys_concatenated

def input(step, t_max, nbr_agent, size_B_concatenated, size_B, type = "step"):
    nbr_step = int(t_max/step)
    if type == "step":
        u_sys = np.ones((size_B, nbr_step))
    elif type == "sinusoidal":
        u_sys = np.cos(np.row_stack((np.arange(0, t_max, step), np.arange(0, t_max, step))))
    elif type == "zero":
         u_sys = np.zeros((size_B, nbr_step))
    u_concatenated = np.reshape(np.array([u_sys for _ in range(nbr_agent)]), (size_B_concatenated, -1))
    return u_sys, u_concatenated

def plot_states(x, x_hat, t_max, step):

    color = ["m", "#FFA500", "#ff6961", "#77DD77"]

    for j in range(x.shape[0]):

        plt.plot(np.arange(0,t_max, step), np.transpose(x_hat[:, j,:]), color[j])    
        plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]), c ="#1E7DF0", ls = "dashed")

        # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 
        plt.title("Step response")


    plt.grid()
    plt.show()

def plot_k(nbr_agent, t_max, step, k_adapt):
    for ind in range(nbr_agent):
        plt.plot(np.arange(0,t_max, step), np.transpose(k_adapt[ind,:]), label = "k("+ str(ind+1)+")")   

    plt.grid()
    plt.show()