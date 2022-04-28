from .parameters_function import *
from .system import *
from harold import staircase
import matplotlib.pyplot as plt
from control import obsv

class ObserverDesign:
    step = 0.01
    def __init__(self, multi_agent_system, t_max, x0, gamma, k0, input = "step", std_noise_parameters = 0):
        self.multi_agent_system = multi_agent_system
        self.std_noise_parameters = std_noise_parameters
        self.gamma = gamma
        self.k0 = k0
        self.t_max = t_max

        print(self.multi_agent_system.tuple_output_matrix)

        self.C_sys_concatenated = diag(self.multi_agent_system.tuple_output_matrix)

        self.A_sys_noisy_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.A_plant)
        if not(self.std_noise_parameters == 0):
            self.multi_agent_system.add_noise(self.std_noise_parameters)
            self.A_sys_noisy_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.A_plant_noisy)
        
        self.B_sys_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.B_plant)

        self.K_sys = np.zeros((np.shape(self.multi_agent_system.B_plant)[1], np.shape(self.multi_agent_system.A_plant)[0]))

        nbr_step = int(self.t_max/self.step)

        self.x = np.zeros((self.multi_agent_system.size_plant, nbr_step))
        self.x[:, 0] = np.transpose(x0)
        self.x_hat = 7*np.random.rand(self.multi_agent_system.nbr_agent, self.multi_agent_system.size_plant, nbr_step)
        
        self.y = np.zeros((np.shape(self.C_sys_concatenated)[0], nbr_step))
        self.y_concatenated = np.zeros((self.multi_agent_system.nbr_agent, nbr_step))
        self.y_hat = np.zeros((self.multi_agent_system.nbr_agent, np.shape(self.C_sys_concatenated)[0], nbr_step))
        self.y_hat_concatenated = np.zeros((self.multi_agent_system.nbr_agent, nbr_step))

        if input == "step":
            self.u_sys = np.ones((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))
        elif input == "cos":
            self.u_sys = np.cos(np.row_stack((np.arange(0, t_max, self.step), np.arange(0, t_max, self.step))))
        elif input == "zero":
            self.u_sys = np.zeros((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))
        
    def parameters(self):
        A_sys = self.multi_agent_system.A_plant
        B_sys = self.multi_agent_system.B_plant
        C = self.multi_agent_system.tuple_output_matrix

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

        k = np.ones(self.multi_agent_system.nbr_agent)

        for i in range(self.multi_agent_system.nbr_agent):
            size_obsv = np.linalg.matrix_rank(obsv(A_sys, C[i]))
            A_bar[str(i)], B_bar[str(i)], C_bar[str(i)], T[str(i)] = staircase(A_sys, B_sys, np.reshape(C[i], (1,np.shape(A_sys)[0])), form = "o")
            Ad[str(i)], _, _ = separate_A_bar(A_bar[str(i)], size_obsv)
            Hd[str(i)] = Hid(C_bar[str(i)], size_obsv)
            Ld[str(i)] = Lid(Ad[str(i)], Hd[str(i)])
            Md[str(i)] = Mid(Ad[str(i)], Ld[str(i)], Hd[str(i)])
            M_dict[str(i)] = Mi(T[str(i)], k[i], Md[str(i)], np.shape(A_sys)[0] - size_obsv)
            L_dict[str(i)] = Li(T[str(i)], np.reshape(Ld[str(i)], (-1,)).T,  np.shape(A_sys)[0] - size_obsv)

        M = [v for v in M_dict.values()]
        M = diag(M)

        L = [v for v in L_dict.values()]
        L = diag(L)

        self.M = M
        self.L = L
        self.T = T
        self.Md = Md
        self.Ld = Ld
        self.M_dict = M_dict
        self.L_dict = L_dict

        return M, L, T, Md, Ld, M_dict, L_dict

    def feedback_control_with_observer(self, desired_eig = None, feedback_gain = None):

        if not(desired_eig == None):
            if self.std_noise_parameters == 0:
                self.K_sys = place(self.multi_agent_system.A_plant_noisy, self.multi_agent_system.B_plant, desired_eig)
            else:
                self.K_sys = place(self.multi_agent_system.A_plant, self.multi_agent_system.B_plant, desired_eig)

        elif not(feedback_gain == None):
            self.K_sys = feedback_gain

    def run_observer(self, type_observer = "output error"):

        nbr_step = int(self.t_max/self.step)
        Laplacien_m = np.kron(self.multi_agent_system.graph.Lap, np.eye(np.shape(self.multi_agent_system.A_plant)[0]))
        x_hat_concatenated = np.reshape(self.x_hat, (self.multi_agent_system.nbr_agent*self.multi_agent_system.size_plant, nbr_step))
        u_concatenated = np.reshape(np.array([self.u_sys for _ in range(self.multi_agent_system.nbr_agent)]), (np.shape(self.B_sys_concatenated )[1], -1))
        K_sys_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.K_sys)
        k_adapt = np.ones((self.multi_agent_system.nbr_agent, nbr_step))
        first = True

        for i in range(nbr_step-1):

            self.x[:,i+1] = self.step*np.dot(self.multi_agent_system.A_plant, self.x[ :,i]) - self.step*np.dot(np.dot(self.multi_agent_system.B_plant, self.K_sys), self.x_hat[0,:,i]) + self.step*np.reshape(np.dot(self.multi_agent_system.B_plant, self.u_sys[:,i]), (-1,)) + self.x[:,i] 
            x_concatenated = np.array([self.x[:, i] for _ in range(self.multi_agent_system.nbr_agent)])
            x_concatenated = np.reshape(x_concatenated, (np.shape(x_concatenated)[0]*np.shape(x_concatenated)[1], ))
            self.y_concatenated[:, i+1] = np.dot(self.C_sys_concatenated, x_concatenated)

            if np.allclose(self.y_concatenated[:, i] - self.y_hat_concatenated[:,i], 0, atol= 10**(-3)) and first and i != 0:
                print(i*self.step)
                first = False

            if type_observer == "output error":
                diff_output = self.y_concatenated[:, i] - self.y_hat_concatenated[:,i]
            elif type_observer == "sliding mode sign":
                diff_output = np.sign(self.y_concatenated[:, i] - self.y_hat_concatenated[:,i])
            elif type_observer == "sliding mode tanh":
                diff_output = np.tanh(10*(self.y_concatenated[:, i] - self.y_hat_concatenated[:,i]))
            else:
                raise Exception("This type of observer doesn't exist, the existing types are: output error, sliding mode sign and sliding mode tanh")

            x_hat_concatenated[:,i+1] = self.step*np.dot(self.A_sys_noisy_concatenated - np.dot(self.B_sys_concatenated, K_sys_concatenated), x_hat_concatenated[:,i]) + self.step*np.reshape(np.dot(self.B_sys_concatenated, u_concatenated[:,i]), (-1, )) + x_hat_concatenated[:,i]  + self.step*np.dot(self.L, diff_output) + self.step*self.gamma*np.dot(np.dot(np.linalg.inv(self.M), -Laplacien_m), x_hat_concatenated[:, i])
            self.x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (self.multi_agent_system.nbr_agent, np.shape(self.multi_agent_system.A_plant)[0],))
            self.y_hat_concatenated[:,i+1] = np.dot(self.C_sys_concatenated, x_hat_concatenated[:,i])

            for ind in range(self.multi_agent_system.nbr_agent):
                k_adapt[ind,i+1] = k_adapt[ind,i] + self.step*np.linalg.norm(np.dot(self.multi_agent_system.graph.Lap[ind],(self.x_hat[:,:, i+1] - self.x_hat[ind,:, i+1])**2), 2)
                size_obsv = np.linalg.matrix_rank(obsv(self.multi_agent_system.A_plant, self.multi_agent_system.tuple_output_matrix[ind]))
                self.M_dict[str(ind)] = Mi(self.T[str(ind)], k_adapt[ind, i+1], self.Md[str(ind)], np.shape(self.multi_agent_system.A_plant)[0] - size_obsv)
                self.L_dict[str(ind)] = Li(self.T[str(ind)], np.reshape(self.Ld[str(ind)], (-1,)).T,  np.shape(self.multi_agent_system.A_plant)[0] - size_obsv)

            self.M = [v for v in self.M_dict.values()]
            self.M = diag(self.M)

            self.L = [v for v in self.L_dict.values()]
            self.L = diag(self.L)


        self.k_adapt = k_adapt


    def plot_states(self, color = ["m", "#FFA500", "#ff6961", "#77DD77"]):

        for j in range(self.x.shape[0]):

            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.x_hat[:, j,:]), color[j])    
            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.x[j,:]), c ="#1E7DF0", ls = "dashed")

            # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 
            plt.title("Step response")


        plt.grid()
        plt.show()

    def plot_k(self):
        for ind in range(self.multi_agent_system.nbr_agent):
            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.k_adapt[ind,:]), label = "k("+ str(ind+1)+")")   

        plt.grid()
        plt.show()