from .parameters_function import *
from .system import *
from harold import staircase
import matplotlib.pyplot as plt

class ObserverDesign:
    """ This class designs "the distributed Luenberger
    observer" and "the distributed finite-time
    observer" for a multi-agent system, controls this
    latter and realises tests using different types of
    inputs.
    The Distribued Luenberger Observer is defined as 
    follows: 
        dx_hat_i = A  x_hat_i + L_i(y_i - C_i x_hat_i) + gamma M_i(k_i)^(-1) sum_{j in x_i' neighbor} Adjacency_ij (x_hat_j - x_hat_i)
        (See article: https://ieeexplore.ieee.org/abstract/document/7799336?casa_token=SbrFD3Nbg9wAAAAA:wb0SGf4Me7eyVsPcYMdnPQPd6gambu2XQImDgxpcZ8lqdiq_Hns4sA6DpPugyb2IYAJmHC5V7oYe )
   
    The Distribued Luenberger Observer is defined as 
    follows: 
        dx_hat_i = A  x_hat_i + L_i[y_i - C_i x_hat_i]^gamma_i + sum_{j in x_i' neighbor} Adjacency_ij [x_hat_j - x_hat_i]^beta
    With:
        [x]^a <=> [sign(x1)*x1^a  sign(x2)*x2^a ... sign(xn)*xn^a]
        (See article: https://ieeexplore.ieee.org/abstract/document/8360487/?casa_token=7c_pyhnc_woAAAAA:TVjPGoTVJDaTLFQ9nnF8PlU6zyNNSl1nlJ6AiNmtBp3Rtutuim0XGzYtS8GVtu_vCkOHMsz3vv4nWA )

    """
    step = 0.01 # The precision of time
    def __init__(self, multi_agent_system, x0, gamma, k0, desired_states= None, x_hat_0 = None, t_max = None, input = "step", std_noise_parameters = 0, std_noise_sensor = 0, std_noise_relative_sensor = 0):
        """ 
        Arguments:
            multi_agent_system: a MultiAgentSystem object for which 
            the observervation will be performed.
            x0: the initial value of the states.
            gamma: a parameter of the observer (its purpose changes
            depending on the type of observer chosen).
            desired_states: the desired state contains all the xd 
            for each agent.
            k0: the initial value of the parameter k_i.
            t_max: the duration of the response and observation.
            input: the type of the input (step, random, cos, zero).
            std_noise_parameters: the standard deviation of the 
            noise added to the parameters of the plant.
            std_noise_sensors: the standard deviation of the noise
            added to the outputs of the system y_i.
            std_noise_relative_sensor: the standard deviation of 
            the noise added to the outputs of the added relative 
            measurements when it is necessary to correct defaults.
        Returns:
            None
        """
        np.random.seed(0)

        self.multi_agent_system = multi_agent_system

        if not(self.multi_agent_system.is_jointly_obsv()):
            raise Exception("This system is not jointly observable")

        self.std_noise_parameters = std_noise_parameters
        self.std_noise_sensor = std_noise_sensor
        self.std_noise_relative_sensor = std_noise_relative_sensor
        self.gamma = gamma
        self.k0 = k0
        self.t_max = t_max
        self.t_response = None

        self.C_sys_concatenated = diag(self.multi_agent_system.tuple_output_matrix)

        self.A_sys_noisy_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.A_plant)
        if not(self.std_noise_parameters == 0):
            self.multi_agent_system.add_noise(self.std_noise_parameters)
            self.A_sys_noisy_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.A_plant_noisy)
        
        self.B_sys_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.multi_agent_system.B_plant)

        self.K_sys = np.zeros((np.shape(self.multi_agent_system.B_plant)[1], np.shape(self.multi_agent_system.A_plant)[0]))

        nbr_step = int(self.t_max/self.step) if t_max != None else 10000

        self.x = np.zeros((self.multi_agent_system.size_plant, nbr_step))
        self.x[:, 0] = np.transpose(x0)

        if np.all(desired_states == None):
            self.xd = np.zeros((self.multi_agent_system.size_plant, nbr_step))
            self.dxd = np.zeros((self.multi_agent_system.size_plant, nbr_step))
            self.xd_concatenated = np.tile(self.xd, (self.multi_agent_system.nbr_agent, 1))
        else:
            self.xd = np.row_stack(desired_states)
            temp = np.column_stack((np.zeros((self.xd.shape[0])), self.xd))
            self.dxd = np.diff(temp)/self.step

            self.xd_concatenated = np.tile(self.xd, (self.multi_agent_system.nbr_agent, 1))

        self.x_hat = np.zeros((self.multi_agent_system.nbr_agent, self.multi_agent_system.size_plant, nbr_step))
        if np.all(x_hat_0 == None):
            self.x_hat[:, : , 0] = np.random.uniform(-10, 10, (self.multi_agent_system.nbr_agent, self.multi_agent_system.size_plant))
        else:
            self.x_hat[:, : , 0] = x_hat_0

        self.y = np.zeros((np.shape(self.C_sys_concatenated)[0], nbr_step))
        self.y_concatenated = np.zeros((np.shape(self.C_sys_concatenated)[0], nbr_step))
        self.y_hat = np.zeros((self.multi_agent_system.nbr_agent, np.shape(self.C_sys_concatenated)[0], nbr_step))
        self.y_hat_concatenated = np.zeros((np.shape(self.C_sys_concatenated)[0], nbr_step))

        if input == "step":
            self.u_sys = np.ones((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))
        if input == "random":
            self.u_sys = np.random.rand(np.shape(self.multi_agent_system.B_plant)[1], nbr_step)
        elif input == "cos":
            self.u_sys = np.zeros((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))
        elif input == "zero":
            self.u_sys = np.zeros((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))
        else:
            self.u_sys = np.expand_dims(np.random.rand(np.shape(self.multi_agent_system.B_plant)[1]), axis = 1)*np.ones((np.shape(self.multi_agent_system.B_plant)[1], nbr_step))

    def parameters(self):
        """ This function choses the right parameters to ensure the
        convergence of the observer
        Arguments:
            None
        Returns:
            M: a matrix containing all the parameter matrices M_i 
            on its diagonal.
            L: a matrix containing all the parameter matrices L_i 
            on its diagonal.
            T: a matrix containing all the transformation matrices
            T_i of each agent on its diagonal.
            Md: a dictionnary containing all the parameter matrices
            M_id.
            Ld: a dictionnary containing all the parameter matrices
            L_id.
            M_dict: a dictionnary containing all the parameter 
            matrices M_i.
            L_dict: a dictionnary containing all the parameter 
            matrices L_i.
        """
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
            A_bar[str(i)], B_bar[str(i)], C_bar[str(i)], T[str(i)] = staircase(A_sys, B_sys, np.reshape(C[i], (-1,np.shape(A_sys)[0])), form = "o")
            Ad[str(i)], _, _ = separate_A_bar(A_bar[str(i)], size_obsv)
            Hd[str(i)] = Hid(C_bar[str(i)], size_obsv)
            Ld[str(i)] = Lid(Ad[str(i)], Hd[str(i)])
            Md[str(i)] = Mid(Ad[str(i)], Ld[str(i)], Hd[str(i)])
            M_dict[str(i)] = Mi(T[str(i)], k[i], Md[str(i)], np.shape(A_sys)[0] - size_obsv)
            L_dict[str(i)] = Li(T[str(i)], Ld[str(i)].T,  np.shape(A_sys)[0] - size_obsv)

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
        """ This function adds a feedback control to the system
        either by choosing the feedback gain or the desired 
        eigen values of the closed loop.
        Arguments:
            desired_eig: the desired eigen values of the closed
            loop.
            feedback_gain: chosen feedback gain.
            noisy: if True, realises feedback control with the 
            noisy plant.
        Returns:
            None
        """
        if not(np.all(desired_eig) == None):
            if self.std_noise_parameters != 0:
                self.K_sys = place(self.multi_agent_system.A_plant_noisy, self.multi_agent_system.B_plant, desired_eig)
            else:
                self.K_sys = place(self.multi_agent_system.A_plant, self.multi_agent_system.B_plant, desired_eig)

        elif not(np.all(feedback_gain) == None):
            self.K_sys = feedback_gain

    def run_observer(self, type_observer = "output error", lost_connexion = None, tol_t_response = 10**(-2)):
        """ This function runs the observation algorithm with the 
        adaptive algorithm for choosing k_i.
        Arguments:
            type_observer: choosing how to write the observation 
            error in the design of the observer, there are four types:
            "output error", "sliding mode sign",  "sliding mode tanh", 
            and "super twisting".
            lost_connexion: it is entered to the function if we wan to
            simulate a loss of connection of some agents
            [[list agents that lost connection], start time of connection loss in seconds, end time of connection loss in seconds]
            tol_t_response: tolerance of the step response.
        Returns:
            the mean of the agents' estimates.
            step response.
        """

        self.parameters()

        nbr_step = int(self.t_max/self.step) if self.t_max != None else 10000
        Laplacien_m = np.kron(self.multi_agent_system.graph.Lap, np.eye(np.shape(self.multi_agent_system.A_plant)[0]))
        x_hat_concatenated = np.reshape(self.x_hat, (self.multi_agent_system.nbr_agent*self.multi_agent_system.size_plant, nbr_step))
        u_concatenated = np.reshape(np.array([self.u_sys for _ in range(self.multi_agent_system.nbr_agent)]), (np.shape(self.B_sys_concatenated )[1], -1))
        K_sys_concatenated = np.kron(np.eye(self.multi_agent_system.nbr_agent), self.K_sys)
        self.k_adapt = np.ones((self.multi_agent_system.nbr_agent, nbr_step))
        self.k_adapt[:, 0] = self.k0
        first = True

        beta = 1

        if np.all(lost_connexion == None):
            lost_connexion = [[], 0, 0]

        w = 0 # initialisation for super twiting

        self.obsv_error_2 = np.zeros(np.shape(self.x))

        for i in range(nbr_step-1):

            self.x[:,i+1] = self.step*np.dot(self.multi_agent_system.A_plant, self.x[ :,i] - self.xd[ :,i]) - self.step*np.dot(np.dot(self.multi_agent_system.B_plant, self.K_sys), np.mean(self.x_hat[:,:,i], axis =0)  - self.xd[ :,i]) + self.step*np.reshape(np.dot(self.multi_agent_system.B_plant, self.u_sys[:,i]), (-1,)) + self.x[:,i] - self.xd[ :,i] + self.dxd[ :,i]*self.step
            x_concatenated = np.array([self.x[:, i] for _ in range(self.multi_agent_system.nbr_agent)])
            x_concatenated = np.reshape(x_concatenated, (np.shape(x_concatenated)[0]*np.shape(x_concatenated)[1], ))
            self.y_concatenated[:, i+1] = np.dot(self.C_sys_concatenated, x_concatenated)

            y_concatenated_noisy = self.add_sensors_noises(self.y_concatenated[:, i]) 

            if i> 20:
                # if np.allclose(np.sum(self.x[:, i-20:i], axis = 1)/20 - np.max(np.mean(self.x_hat[:, :, i-20:i], axis = 0), axis = 1), 0, atol= tol_t_response) and first:
                if np.allclose(np.sum(self.y_hat_concatenated[:,i-20:i], axis = 1)/20 - np.max(self.y_hat_concatenated[:,i-20:i], axis = 1), 0, atol= tol_t_response) and first:
                    print(i*self.step, "s")
                    self.t_response = i*self.step
                    if self.t_max == None:
                        self.t_max = i*self.step
                        nbr_step = int(self.t_max/self.step) 
                        self.x_hat = self.x_hat[:, : , :nbr_step]
                        self.x = self.x[: , :nbr_step]

                        self.static_error = np.abs(self.y_concatenated[:, i] - self.y_hat_concatenated[:,i])
                        return np.mean(self.x_hat, axis= 0)[:, -1], i*self.step
                    first = False

            if type_observer == "DLO":
                diff_output = y_concatenated_noisy - self.y_hat_concatenated[:,i] 
                consensus_tot =  self.gamma*np.dot(np.linalg.inv(self.M), np.dot(-Laplacien_m, x_hat_concatenated[:, i] - self.xd_concatenated[:, i]))
            
            elif type_observer == "sliding mode sign":
                diff_output = np.sign(y_concatenated_noisy - self.y_hat_concatenated[:,i])
                consensus_tot = self.gamma*np.dot(np.linalg.inv(self.M), np.dot(-Laplacien_m, x_hat_concatenated[:, i] - self.xd_concatenated[:, i]))
            
            elif type_observer == "sliding mode tanh":
                diff_output = np.tanh(10*(y_concatenated_noisy - self.y_hat_concatenated[:,i]))
                consensus_tot = self.gamma*np.dot(np.linalg.inv(self.M), np.dot(-Laplacien_m, x_hat_concatenated[:, i] - self.xd_concatenated[:, i]))
            
            elif type_observer == "super twisting":
                diff_output = np.tanh(20*(y_concatenated_noisy - self.y_hat_concatenated[:,i]))*np.abs(y_concatenated_noisy - self.y_hat_concatenated[:,i])**(4) + 0*w
                w += self.step*10*np.tanh(10*(y_concatenated_noisy - self.y_hat_concatenated[:,i]))
                consensus_tot = self.gamma*np.dot(np.linalg.inv(self.M), np.dot(-Laplacien_m, x_hat_concatenated[:, i] - self.xd_concatenated[:, i]))
            
            elif type_observer == "DFTO":
                gamma_i = (1+self.gamma)/(1 + (self.multi_agent_system.size_plant - 1)*self.gamma)
                beta = 1 + self.gamma

                diff_output = np.sign((y_concatenated_noisy - self.y_hat_concatenated[:,i]))*np.abs(y_concatenated_noisy - self.y_hat_concatenated[:,i])**(gamma_i)
                
                consensus = 0
                for l in range(self.multi_agent_system.nbr_agent):
                    for neighbor in np.where(self.multi_agent_system.graph.Adj[l,:]>0)[0]:
                        consensus += np.sign((self.x_hat[l,:, i+1] - self.x_hat[neighbor,:, i+1]))*(np.abs(self.x_hat[l,:, i+1] - self.x_hat[neighbor,:, i+1]))**(beta)
                
                consensus_tot = np.ndarray.flatten(np.array([consensus for _ in range(self.multi_agent_system.nbr_agent)]))
                
            else: 
                raise Exception("This type of observer doesn't exist, the existing types are: DLO, DFTO, sliding mode sign and sliding mode tanh")

            x_hat_concatenated[:,i+1] = self.step*np.dot(self.A_sys_noisy_concatenated - np.dot(self.B_sys_concatenated, K_sys_concatenated), x_hat_concatenated[:,i] - self.xd_concatenated[:, i]) + self.step*np.reshape(np.dot(self.B_sys_concatenated, u_concatenated[:,i]), (-1, )) + x_hat_concatenated[:,i] - self.xd_concatenated[:, i]  + self.step*np.dot(self.L, diff_output) + self.step*consensus_tot  + np.tile(self.dxd[ :,i], (self.multi_agent_system.nbr_agent))*self.step

            if (i>lost_connexion[1]/self.step and i<lost_connexion[2]/self.step):
                for agent in lost_connexion[0]:
                    min_i = int(agent*self.multi_agent_system.A_plant.shape[0])
                    max_i = int((agent+1)*self.multi_agent_system.A_plant.shape[0])

                    min_u = int(agent*self.multi_agent_system.B_plant.shape[1])
                    max_u = int((agent+1)*self.multi_agent_system.B_plant.shape[1])


                    # x_hat_concatenated[min_i:max_i,i+1] = x_hat_concatenated[min_i:max_i,i]
                    x_hat_concatenated[min_i:max_i, i+1] = self.step*np.dot(self.multi_agent_system.A_plant - np.dot(self.multi_agent_system.B_plant, self.K_sys), x_hat_concatenated[min_i:max_i, i]) + self.step*np.reshape(np.dot(self.multi_agent_system.B_plant, u_concatenated[min_u:max_u,i]), (-1, )) + x_hat_concatenated[min_i:max_i,i]  
            
            
            self.x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (self.multi_agent_system.nbr_agent, np.shape(self.multi_agent_system.A_plant)[0],))
            self.y_hat_concatenated[:,i+1] = np.dot(self.C_sys_concatenated, x_hat_concatenated[:,i])

            self.obsv_error_2[:, i+1] = self.obsv_error_2[:, i] + self.step*(self.x[:, i] - np.mean(self.x_hat[:, :, i], axis = 0))**2
            
            for ind in range(self.multi_agent_system.nbr_agent):
                self.k_adapt[ind,i+1] = self.k_adapt[ind,i] + self.step*np.linalg.norm(np.dot(self.multi_agent_system.graph.Lap[ind],np.abs(self.x_hat[:,:, i+1] - self.x_hat[ind,:, i+1])**2), 2) 
                size_obsv = np.linalg.matrix_rank(obsv(self.multi_agent_system.A_plant, self.multi_agent_system.tuple_output_matrix[ind]))
                self.M_dict[str(ind)] = Mi(self.T[str(ind)], self.k_adapt[ind, i+1], self.Md[str(ind)], np.shape(self.multi_agent_system.A_plant)[0] - size_obsv)
                self.L_dict[str(ind)] = Li(self.T[str(ind)], self.Ld[str(ind)].T,  np.shape(self.multi_agent_system.A_plant)[0] - size_obsv)

            self.M = [v for v in self.M_dict.values()]
            self.M = diag(self.M)

            self.L = [v for v in self.L_dict.values()]
            self.L = diag(self.L)

        self.x_hat +=  self.xd
        print("k", self.k_adapt[:, i])
        
        return 0, 0

    def add_sensors_noises(self, y_concatenated):
        """ This function adds noise to the output of the system, this
        is equivalent to adding noise to sensors.
        Arguments:
            y_concatenated: the output to which noise will be added.
        Returns:
            the noisy version of the output y.
        """

        if self.multi_agent_system.type == "MultiAgentGroups":
            sigma = self.std_noise_relative_sensor*self.multi_agent_system.added_output
            y_concatenated_noisy = y_concatenated + np.random.normal(0, self.std_noise_sensor, np.shape(y_concatenated))
        
            y_concatenated_noisy += gaussian_noise(np.zeros(np.shape(y_concatenated)), sigma, np.shape(y_concatenated))

            return y_concatenated_noisy
        
        else:
            return y_concatenated + np.random.normal(0, self.std_noise_sensor, np.shape(y_concatenated))

    def plot_states(self, saveFile = None, color = ["m", "#FFA500", "#ff6961", "#77DD77", "#5CA0FF", "#FFF35A", "#762EFF", "#5AECFF", "#00E02D", "#B10FFF"]):
        """ This function plots the response of the system to its 
        input and the estimates of the observer.
        Arguments:
            saveFile: the directory where to save the plot, if None
            show the plot without saving.
            color: the color of the plot of the estimates of the 
            states. One color for each agent.
        Returns:
            None
        """
        for j in range(self.x.shape[0]):

            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.x_hat[:,j ,:]), color[j%len(color)])    
            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.x[j,:]), c ="#1E7DF0", ls = "dashed")
            plt.xlabel("time")
            plt.ylabel("x(t) and its estimates")
            # plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.xd[j,:]), c= "#000000")
            # plt.title("Step response")


        plt.grid()
        if saveFile != None:
            plt.savefig(saveFile + "StateAndEstimate" + "param" + str(self.std_noise_parameters*100) + "sensor" + str(self.std_noise_sensor) + ".png", dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_k(self, saveFile = None):
        """ This function plots the parameters k_i at all times.
        Arguments:
            saveFile: the directory where to save the plot, if None
            show the plot without saving.
        Returns:
            None
        """
        nbr_step = int(self.t_max/self.step) if self.t_max != None else 10000
        for ind in range(self.multi_agent_system.nbr_agent):
            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.k_adapt[ind,: nbr_step]), label = "k("+ str(ind+1)+")")   

        plt.grid()
        if saveFile != None:
            plt.savefig(saveFile + "self.k_adapt" + "param" + str(self.std_noise_parameters*100) + "sensor" + str(self.std_noise_sensor) + ".png", dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_criateria(self, saveFile = None):
        """ This function plots the criterion (x_i- x_hat_i)^2.
        Arguments:
            saveFile: the directory where to save the plot, if None
            show the plot without saving.
        Returns:
            None
        """

        nbr_step = int(self.t_max/self.step) if self.t_max != None else 10000

        plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.obsv_error_2[:, :nbr_step])) 
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("Integral error squared")
        # plt.title("observation error")
        if saveFile != None:
            plt.savefig(saveFile  + "ObsvError2" + "param" + str(self.std_noise_parameters*100) + "sensor" + str(self.std_noise_sensor) + ".png", dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_obsv_error(self, saveFile = None, color = ["m", "#FFA500", "#ff6961", "#77DD77", "#5CA0FF", "#FFF35A", "#762EFF", "#5AECFF", "#00E02D", "#B10FFF"]):
        """ This function plots the observation error x_i- x_hat_i
        Arguments:
            saveFile: the directory where to save the plot, if None
            show the plot without saving.
            color: the color of the plot of the error of the states.
            One color for each agent.
        Returns:
            None
        """
        for j in range(self.x.shape[0]):
            plt.plot(np.arange(0,self.t_max, self.step), np.transpose(self.x[j,:]-self.x_hat[:,j,:]), color[j%len(color)]) 
            # plt.title("Observation error y - y_estimate")


        plt.grid()
        if saveFile != None:
            plt.savefig(saveFile  + "obsvError"+ "param" + str(self.std_noise_parameters*100) + "sensor" + str(self.std_noise_sensor) + ".png", dpi=150)
            plt.close()
        else:
            plt.show()


class CooperativeObserverDesign(ObserverDesign):
    step = 0.01 # The precision of time
    # THIS ONLY WORKS FOR SYSTEMS WITH SAME SIZES
    def __init__(self, heterogeneous_MAS, x0, gamma, k0, x_hat_0 = None, t_max = None, input = "step", std_noise_parameters = 0, std_noise_sensor = 0, std_noise_relative_sensor = 0):
    
        np.random.seed(0)

        self.heterogeanous_MAS = heterogeneous_MAS
        self.MAS_list = heterogeneous_MAS.MAS_list

        self.std_noise_parameters = std_noise_parameters
        self.std_noise_sensor = std_noise_sensor
        self.std_noise_relative_sensor = std_noise_relative_sensor
        self.gamma = gamma
        self.k0 = k0
        self.t_max = t_max
        self.t_response = None

        nbr_step = int(self.t_max/self.step) if t_max != None else 10000
        
        self.x_list = list([])
        self.x_hat_list = list([])
        self.x_hat_flat_list = list([])

        self.y_list = list([])
        self.y_hat_list = list([])
        self.y_hat_flat_list = list([])

        for i in range(len(self.heterogeanous_MAS.agent_in_neighborhood)):
            self.x_list.append(np.zeros((self.heterogeanous_MAS.main_system_list[i][0].shape[0], nbr_step)))
            self.x_hat_list.append(np.zeros((len(self.heterogeanous_MAS.agent_in_neighborhood[i]), self.heterogeanous_MAS.main_system_list[i][0].shape[0], nbr_step)))
            self.x_hat_flat_list.append(np.zeros((len(self.heterogeanous_MAS.agent_in_neighborhood[i])*self.heterogeanous_MAS.main_system_list[i][0].shape[0], nbr_step)))

            self.y_list.append(np.zeros((self.heterogeanous_MAS.main_system_list[i][2].shape[0], nbr_step)))
            self.y_hat_list.append(np.zeros((len(self.heterogeanous_MAS.agent_in_neighborhood[i]),self.heterogeanous_MAS.main_system_list[i][2].shape[0], nbr_step)))
            self.y_hat_flat_list.append(np.zeros((len(self.heterogeanous_MAS.agent_in_neighborhood[i]),self.heterogeanous_MAS.main_system_list[i][2].shape[0], nbr_step)))

   
    def run_observer(self, type_observer = "output error", lost_connexion = None, tol_t_response = 10**(-2)):

        nbr_step = int(self.t_max/self.step) if self.t_max != None else 10000

        for k in range(len(self.heterogeanous_MAS.agent_in_neighborhood)):
            for i in range(nbr_step-1):
                self.x_list[k][:, i+1] = self.step*np.dot(self.heterogeanous_MAS.main_system_list[k][0], self.x_list[k][ :,i]) - 0 + self.step*np.reshape(np.dot(self.heterogeanous_MAS.main_system_list[k][1], np.ones((self.heterogeanous_MAS.main_system_list[k][1].shape[1], 1))), (-1,)) + self.x_list[k][:,i] 
                x_concatenated = np.ndarray.flatten(np.array(self.x_list)[self.heterogeanous_MAS.agent_in_neighborhood[k]][:, :, i+1])
                self.y_list[k][:, i+1] = np.dot(self.MAS_list[k].tuple_output_matrix, x_concatenated)
        
                L = np.ones((self.x_hat_flat_list[k][:, i+1].shape[0], self.y_list[k][:, i+1].shape[0]))
                K = 0.1*np.ones((self.x_hat_flat_list[k][:, i+1].shape[0], self.x_hat_flat_list[0][0:self.heterogeanous_MAS.main_system_list[0][0].shape[0], i].shape[0]))
                sum_consensus = 0
                for l in range(len(self.heterogeanous_MAS.agent_in_neighborhood)):
                    if k in self.heterogeanous_MAS.agent_in_neighborhood[l] and k!= l:
                        ind = np.where(self.heterogeanous_MAS.agent_in_neighborhood[l] == k)[0][0]
                        sum_consensus += self.x_hat_flat_list[l][ind*(self.heterogeanous_MAS.main_system_list[k][0]).shape[0]:(ind+1)*self.heterogeanous_MAS.main_system_list[k][0].shape[0], i] - self.x_hat_flat_list[l][k*(self.heterogeanous_MAS.main_system_list[k][0]).shape[0]:(k+1)*self.heterogeanous_MAS.main_system_list[k][0].shape[0], i]
                
            
                self.x_hat_flat_list[k][:, i+1] = self.step*np.dot(self.MAS_list[k].A_plant, self.x_hat_flat_list[k][ :,i]) - 0 + self.step*np.reshape(np.dot(self.MAS_list[k].B_plant, np.ones((self.MAS_list[k].B_plant.shape[1], 1))), (-1,)) + np.dot(L, (self.y_list[k][:, i] - np.dot(self.MAS_list[k].tuple_output_matrix, self.x_hat_flat_list[k][:, i]))) +  np.dot(K, sum_consensus) + self.x_hat_flat_list[k][:,i] 

            plt.plot(np.arange(0, self.t_max, self.step), np.transpose(self.y_list[k]))
            plt.plot(np.arange(0, self.t_max, self.step), np.transpose(self.x_hat_flat_list[k]))
            plt.grid()
            plt.show()