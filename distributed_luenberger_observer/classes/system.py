import numpy as np
from .helper_function import *
from control import obsv
from control import place
import matplotlib.pyplot as plt
from harold import staircase
 
class MultiAgentSystem:
    """ This class defines a multi agent system and helps 
    in reading its proprieties, controlling the system, 
    and testing it with different type of inputs
    """
    step = 0.01  # The precision of time

    def __init__(self, A_plant, B_plant, tuple_output_matrix, graph, form = "normal"):
        """ 
        Arguments:
            A_plant: the state space matrix of the entire plant.
            B_plant: the input matrix of the entire plant.
            tuple_output_matrix: a tuple or a dictionnary containing 
            the output matrices of each agent.
        Returns:
            None
        """
        self.size_plant = np.shape(A_plant)[0]
        self.A_plant = A_plant.astype(np.float64)
        self.B_plant = B_plant
        self.tuple_output_matrix = tuple_output_matrix
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)
        self.form = form # The form in which the matrix was written
        self.type = "MultiAgentSystem"
        self.graph = graph
        self.nbr_agent = self.graph.nbr_agent

    def is_jointly_obsv(self):
        """ This function tests if the system is jointly observable
        Arguments:
            None
        Returns:
            True if the system is jointly observable, otherwize False
        """
        return np.linalg.matrix_rank(obsv(self.A_plant, np.row_stack(self.tuple_output_matrix))) == self.size_plant

    def obsv_index(self):
        """ This function tests if the observability of each agent with respect
        to the plant
        Arguments:
            None
        Returns:
            indew_array: a list of the observability index of each agent with 
            respect to the plant order following the argument self.tuple_output_matrix
        """
        index_array = np.zeros((self.nbr_agent,))
        for i in range(self.nbr_agent):
            index_array[i] = np.linalg.matrix_rank(obsv(self.A_plant, self.tuple_output_matrix[i]))
        return index_array

    def step_response(self, t_max, x0 = None, noisy = False, stabilized = False):
        """ This function simulates the step response of the system
        Arguments:
            t_max: the duration of the response.
            x0: the initial values of the states.
            noisy: if True, the test is performed with the noisy 
            matrix' plant.
            stabilized: the step response of the system after 
            stabilization.
        Returns:
            None
        """

        if noisy:
            A = self.A_plant_noisy
        if stabilized:
            A = self.A_plant_stabilized
        if not(noisy) and not(stabilized):
            A = self.A_plant
        
        if x0 == None:
            x0 = np.zeros((self.size_plant,))

        if not(np.shape(x0)[0] == self.size_plant):
            raise Exception("Size of the plant and the initial value don't match")

        x = np.zeros((self.size_plant, int(t_max/self.step)))
        x[:, 0] = x0

        for i in range(int(t_max/self.step-1)):
            x[:,i+1] = self.step*np.dot(A, x[ :,i]) + self.step*np.reshape(np.dot(self.B_plant, np.ones((np.shape(self.B_plant)[1], 1))), (-1,)) + x[:,i] 

        t = np.arange(0, t_max, self.step)
        x = np.transpose(x)

        plt.plot(t, x[:, 0])
        plt.plot(t, x[:, 1])
        plt.plot(t, np.ones(np.shape(t)), "b--")
        plt.legend(["x1(t)", "x2(t)", "step"], loc ="lower right")
        plt.title("step response")
        plt.xlabel("time")
        plt.ylabel("x(t)")
        plt.grid()
        plt.show()
         
        

        return t, x

            
    def impulse_response(self, t_max, x0 = None, noisy = False, stabilized = False):
        """ This function simulates the impulse response of the system
        Arguments:
            t_max: the duration of the response.
            x0: the initial values of the states.
            noisy: if True, the test is performed with the noisy 
            matrix' plant.
            stabilized: the impulse response of the system after 
            stabilization.
        Returns:
            None
        """

        if noisy:
            A = self.A_plant_noisy
        if stabilized:
            A = self.A_plant_stabilized
        if not(noisy) and not(stabilized):
            A = self.A_plant

        if x0 == None:
            x0 = np.zeros((self.size_plant,))

        if not(np.shape(x0)[0] == self.size_plant):
            raise Exception("Size of the plant and the initial value don't match")

        x = np.zeros((self.size_plant, int(t_max/self.step)))
        x[:, 0] = x0

        for i in range(int(t_max/self.step-1)):
            if i == 0:
                x[:,i+1] = self.step*np.dot(A, x[ :,i]) + self.step*np.reshape(np.dot(self.B_plant, np.ones((np.shape(self.B_plant)[1], 1))), (-1,)) + x[:,i] 
            else: 
                x[:,i+1] = self.step*np.dot(A, x[ :,i]) + x[:,i] 
        
        t = np.arange(0, t_max, self.step)
        x = np.transpose(x)

        plt.plot(t,x)
        plt.plot(t[1:], t[1:]*0, "b--")
        plt.scatter(0, 1, color = 'b')
        plt.title("impulse response")
        plt.grid()
        plt.show()

        return t, x

    def forced_response(self, t_max, u, x0 = None, noisy = False, stabilized = False):
        """ This function simulates the response of the system to
        a given input.
        Arguments:
            t_max: the duration of the response.
            u: an array containing the input of the system at each
            step.
            x0: the initial values of the states.
            noisy: if True, the test is performed with the noisy 
            matrix' plant.
            stabilized: the step response of the system after 
            stabilization.
        Returns:
            None
        """

        step = t_max/np.shape(u)[1]

        if noisy:
            A = self.A_plant_noisy
        if stabilized:
            A = self.A_plant_stabilized
        if not(noisy) and not(stabilized):
            A = self.A_plant

        if x0 == None:
            x0 = np.zeros((self.size_plant,))

        if not(np.shape(x0)[0] == self.size_plant):
            raise Exception("Size of the plant and the initial value don't match")

        x = np.zeros((self.size_plant, int(t_max/step)))
        x[:, 0] = x0

        u = np.reshape(u, (-1, int(t_max/step)))

        for i in range(int(t_max/step-1)):
            x[:,i+1] = step*np.dot(A, x[ :,i]) + step*np.reshape(np.dot(self.B_plant, u[:,i]), (-1,)) + x[:,i] 
        
        t = np.arange(0, t_max, step)
        x = np.transpose(x)

        plt.plot(t,x)
        plt.plot(t, np.transpose(u), "--")
        plt.title("Forced response")
        plt.grid()
        plt.show()

        return t, x  

    def feedback_control(self, desired_eig = None, feedback_gain = None, noisy = False):
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
            if not(noisy):
                K = place(self.A_plant_noisy, self.B_plant, desired_eig)
                self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, K)
            else:
                K = place(self.A_plant, self.B_plant, desired_eig)
                self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, K)
            return K

        elif not(np.all(feedback_gain) == None):
            self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, feedback_gain)
            return np.linalg.eig(self.A_plant_stabilized)

            

    def add_noise(self, std_noise):
        """ This function add noise to the plant according to
        std_noise
        Arguments:
            std_noise: the standard deviation of the noise 
            added to the plant.
        Returns:
            None
        """
        if self.form == "standard":
            self.A_plant_noisy = np.array(self.A_plant)
            self.A_plant_noisy[-1] += np.random.normal(0, std_noise*np.abs(np.mean(self.A_plant[-1])), np.shape(self.A_plant[-1]))
        else:
            self.A_plant_noisy = np.array(self.A_plant)
            self.A_plant_noisy += np.random.normal(0, std_noise*np.abs(np.mean(self.A_plant[self.A_plant!=0])), np.shape(self.A_plant))
            # self.A_plant_noisy += std_noise*np.abs(np.mean(self.A_plant[self.A_plant!=0]))
            self.A_plant_noisy[np.where(self.A_plant == 0)] = 0
        
        return self.A_plant_noisy

        


class RandomStandardSystem(MultiAgentSystem):
    """ This class inherits from the multi agent system
    class, it creates a random multi agent system in a 
    standard form.
    """
    step = 0.01  # The precision of time
    def __init__(self, size_plant, random_range, B_plant, tuple_output_matrix, graph):

        self.size_plant = size_plant

        self.A_plant = np.zeros((size_plant,size_plant))
        self.A_plant[0:size_plant-1,1:] = np.eye(size_plant-1)
        self.A_plant[size_plant-1:size_plant,0:] = -random_range*np.abs(np.random.rand(1, size_plant))
        self.A_plant = self.A_plant.astype(np.float64)

    
        self.B_plant = B_plant
        self.tuple_output_matrix = tuple_output_matrix
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)  

        self.form = "standard" 
        self.type = "RandomStandardSystem"

        self.graph = graph
        self.nbr_agent = self.graph.nbr_agent

class MultiAgentGroups(MultiAgentSystem):
    """ This class inherits from the multi agent system
    class, it creates a multi agent system from a group
    of systems.
    """
    step = 0.01  # The precision of time
    def __init__(self, A_agent, B_agent, tuple_output_matrix, graph):
        """ 
        Arguments:
            A_agent: the state space matrix of an agent of the plant
            B_plant: the input matrix of an agent of the plant
            tuple_output_matrix: a tuple or a dictionnary containing 
            the output matrices of each agent.
        Returns:
            None
        """
        self.nbr_agent = graph.nbr_agent
        self.A_plant = np.kron(np.eye(self.nbr_agent), A_agent)
        self.size_plant = np.shape(self.A_plant)[0]
        self.B_plant = np.kron(np.eye(self.nbr_agent), B_agent)
        self.tuple_output_matrix = list()
        for i in range(self.nbr_agent):
            self.tuple_output_matrix.append(np.kron(np.eye(self.nbr_agent)[i], tuple_output_matrix[i][~np.all(tuple_output_matrix[i] == 0, axis=1)]))
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)  

        self.form = "normal" 
        self.type = "MultiAgentGroups"

        self.graph = graph

        self.obsv_index()
        if len(self.find_faulty_agents()) != 0:
            print("There are agents with faults")

        self.added_output = np.zeros(np.shape(np.row_stack(tuple_output_matrix))[0])


    def is_jointly_obsv(self):
        """ This function tests if the system is jointly observable
        Arguments:
            None
        Returns:
            True if the system is jointly observable, otherwize False
        """
        return np.linalg.matrix_rank(obsv(self.A_plant, np.row_stack(self.tuple_output_matrix))) == self.size_plant
        
    def obsv_index(self):
        """ This function tests if the observability of each agent with respect
        to the plant
        Arguments:
            None
        Returns:
            indew_array: a list of the observability index of each agent with 
            respect to the plant order following the argument self.tuple_output_matrix
        """
        self.index_array = np.zeros((self.nbr_agent,))
        for i in range(self.nbr_agent):
            self.index_array[i] = np.linalg.matrix_rank(obsv(self.A_plant[:int(self.size_plant/self.nbr_agent),:int(self.size_plant/self.nbr_agent)], self.tuple_output_matrix[i][:, i*(int(self.size_plant/self.nbr_agent)):(i+1)*int(self.size_plant/self.nbr_agent)]))
        return self.index_array

    def find_faulty_agents(self):
        """ This function find all the agents that are not observable
        to the plant
        Arguments:
            None
        Returns:
            faulty_agents: a list of the positions of the faulty 
            agents.
        """
        self.faulty_agents = np.where(self.index_array < self.size_plant/self.nbr_agent)[0]
        return self.faulty_agents

    def get_needed_states_for_faulty_agent(self, nbr_iteration = 5):
        """ Find the states missing to make an unobservable 
        agent's system observable. It changes the values of
        the output matrices of the agents.
            Arguments:
                nbr_iteration: the number of itertion to 
                run the algorithm before saying that the 
                system can't become observable.
            Returns:
                None
            """
        
        if len(self.faulty_agents) == 0:
            self.added_output = np.zeros(np.row_stack(self.tuple_output_matrix).shape[0])
            print("There are no faulty agents, no need to change the output matrices")
        else:
            
            added_states = 0
            self.added_output = np.array([])

            for i in self.faulty_agents:

                A = self.A_plant[:int(self.size_plant/self.nbr_agent), :int(self.size_plant/self.nbr_agent)]
                C = self.tuple_output_matrix[i][:, i*(int(self.size_plant/self.nbr_agent)):(i+1)*int(self.size_plant/self.nbr_agent)]
                obsv_ind = np.linalg.matrix_rank(obsv(A, C))

                it = 0
                added_obs_states = np.zeros((A.shape[0] - obsv_ind, np.shape(C)[1]))

                while obsv_ind!= A.shape[0] and it < nbr_iteration:
                    # The non-observable states are defined by taking the last lines of the
                    # transformation (obsv/no-obsv) matrix because it will organise them in 
                    # obsv then non-obsv and we need non-obsv
                    _, _, _, T = staircase(A, self.B_plant[:int(self.B_plant.shape[0]/self.nbr_agent), :int(self.B_plant.shape[0]/self.nbr_agent)], C, form = "o")
                    obsv_ind = np.linalg.matrix_rank(obsv(A, C))
                    print("obsv_ind", obsv_ind)

                    # P = np.zeros(A.shape)
                    # P[obsv_ind:, :] = T[obsv_ind:, :]
                    T = T.T
                    # print("transformation matrix", T)
                    added_obs_states = added_obs_states + T[obsv_ind:, :]
                    
                    obsv_ind = np.linalg.matrix_rank(obsv(A, np.row_stack((C, added_obs_states))))
                    it += 1

                if obsv_ind != A.shape[0]:
                    raise Exception("It is not possible for the system to become observable")
                
                # We only need 1 and 0 in the observable states that we need to add 
                # because having real values can consume a lot of memory and it is 
                # not necessary
                added_obs_states = np.float16(np.logical_or(added_obs_states> 10**(-6), added_obs_states<- 10**(-6))) 

                
                for k in range(len(self.tuple_output_matrix)):
                    if (k not in self.faulty_agents) and (k in np.where(self.graph.Adj[i,:]>0)[0]):
                        # Stacking the output matrix of the system and the added relative 
                        # measurements.  
                        shape_before = self.tuple_output_matrix[k].shape[0] 
                        self.tuple_output_matrix[k] = np.row_stack((self.tuple_output_matrix[k],
                                                                    np.kron(
                                                                        np.eye(self.nbr_agent)[i],
                                                                         -added_obs_states)
                                                                    + np.kron(
                                                                        np.eye(self.nbr_agent)[k],
                                                                         added_obs_states)))

                    
                    # This is a way of keeping which states were added and which ones 
                    # were there from the begining, it is useful when adding the noise
                    # to relative sensing.
                    if added_states == len(self.faulty_agents)-1 and (k not in self.faulty_agents):
                        x = np.ones(np.shape(self.tuple_output_matrix[k])[0])
                        x[:shape_before] = 0
                        self.added_output = np.append(self.added_output, x)
                    elif added_states == len(self.faulty_agents)-1 and (k in self.faulty_agents):
                        self.added_output = np.append(self.added_output, np.zeros(np.shape(self.tuple_output_matrix[k])[0]))
        
                added_states += 1
                # self.tuple_output_matrix[i] = np.kron(np.eye(self.nbr_agent)[i], C)

           
        
    def print_plant_state_space(self):
        """ This function prints the states of the multi-
        agent system.
        """
        print("A\n", self.A_plant)
        print("")
        print("B\n", self.B_plant)
        print("")
        print("C\n", np.row_stack(self.tuple_output_matrix))



class HeterogeneousMultiAgentGroups(MultiAgentGroups):
    """ This class inherits from the multi agent groups
    class, it creates a heterogeneous multi agent system
    """
    step = 0.01  # The precision of time
    def __init__(self, tuple_A_agents, tuple_B_agents, tuple_output_matrix, graph):
        """ 
        Arguments:
            A_agent: the state space matrix of an agent of the plant
            B_plant: the input matrix of an agent of the plant
            tuple_output_matrix: a tuple or a dictionnary containing 
            the output matrices of each agent.
        Returns:
            None
        """
        self.nbr_agent = graph.nbr_agent
        self.A_plant = diag(tuple_A_agents)
        self.tuple_A_agents = tuple_A_agents
        self.size_plant = np.shape(self.A_plant)[0]
        self.B_plant = diag(tuple_B_agents)
        self.tuple_B_agents = tuple_B_agents
        self.tuple_output_matrix = tuple_output_matrix
        
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)  

        self.form = "normal" 
        self.type = "HeterogeneousMultiAgentGroups"

        self.graph = graph

        B_plant_list = list([])
        A_plant_list = list([])
        self.MAS_list = list([])
        self.agent_in_neighborhood = list([])
        self.main_system_list = list([])
        for i in range(self.nbr_agent):
            tuple_A = list([])
            tuple_B = list([])
            self.size_agents = list([])
            self.main_system_list.append((self.tuple_A_agents[i], self.tuple_B_agents[i], self.tuple_output_matrix[i][i]))
            for v in np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]:
                tuple_A.append(self.tuple_A_agents[v])
                tuple_B.append(self.tuple_B_agents[v])
                
                self.size_agents.append(self.tuple_A_agents[v].shape[0])

            A_plant_list.append(diag(np.array(self.tuple_A_agents)[np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]]))
            B_plant_list.append(diag(np.array(self.tuple_B_agents)[np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]]))
            C_plants = list([])
            for v in range(np.shape(np.array(self.tuple_output_matrix[i])[np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]])[1]):
                C_plants.append(np.ndarray.flatten(np.array(self.tuple_output_matrix[i])[np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]][:, v, :]))

            neighborhood = np.where((self.graph.Adj[i,:] + np.eye(self.nbr_agent)[i])>0)[0]
            self.agent_in_neighborhood.append(neighborhood)

            A_plant_list.append(diag(tuple_A))
            B_plant_list.append(diag(tuple_B))
            C_plants = np.row_stack(C_plants)

            self.MAS_list.append(MultiAgentSystem(diag(tuple_A), diag(tuple_B), C_plants, self.graph.find_sub_graph(neighborhood)))



# controllability/ consensus