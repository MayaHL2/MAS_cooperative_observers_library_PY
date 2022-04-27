import numpy as np
from helper_function import *
from control import obsv, place
import matplotlib.pyplot as plt
 
class MultiAgentSystem:
    step = 0.01
    e = 10**(-6)
    def __init__(self, nbr_agent, A_plant, B_plant, tuple_output_matrix, graph):
        self.size_plant = np.shape(A_plant)[0]
        self.nbr_agent = nbr_agent
        self.A_plant = A_plant.astype(np.float64)
        self.B_plant = B_plant
        self.tuple_output_matrix = tuple_output_matrix
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)
        self.form = "normal"
        self.graph = graph

    def is_jointly_obsv(self):
        return np.linalg.matrix_rank(obsv(self.A_plant, np.row_stack(self.tuple_output_matrix))) == self.size_plant

    def obsv_index(self):
        index_array = np.zeros((self.nbr_agent,))
        for i in range(self.nbr_agent):
            index_array[i] = np.linalg.matrix_rank(obsv(self.A_plant, self.tuple_output_matrix[i]))
        return index_array

    def step_response(self, t_max, x0 = None, noisy = False, stabilized = False):

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

        plt.plot(t, x)
        plt.plot(t, np.ones(np.shape(t)), "b--")
        plt.title("step response")
        plt.grid()
        plt.show()

        return t, x

            
    def impulse_response(self, t_max, x0 = None, noisy = False, stabilized = False):

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

    def forced_response(self, t_max, step, u, x0 = None, noisy = False, stabilized = False):

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
        # if noisy:
        #     self.add_noise()

        if not(desired_eig == None):
            if not(noisy):
                K = place(self.A_plant_noisy, self.B_plant, desired_eig)
                self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, K)
            else:
                K = place(self.A_plant, self.B_plant, desired_eig)
                self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, K)
            return K

        elif not(feedback_gain == None):
            self.A_plant_stabilized = self.A_plant - np.dot(self.B_plant, feedback_gain)
            return np.linalg.eig(self.A_plant_stabilized)

            

    def add_noise(self, std_noise):
        if self.form == "standard":
            self.A_plant_noisy = np.array(self.A_plant)
            self.A_plant_noisy[-1] = np.random.normal(np.mean(self.A_plant[-1]), std_noise, np.shape(self.A_plant[-1]))
        else:
            self.A_plant_noisy = np.array(self.A_plant)
            self.A_plant_noisy += np.random.normal(0, std_noise, np.shape(self.A_plant))
            self.A_plant_noisy[np.where(self.A_plant == 0)] = 0
        
        return self.A_plant_noisy
        


class RandomStandardSystem(MultiAgentSystem):
    
    def __init__(self, nbr_agent, size_plant, random_range, B_plant, tuple_output_matrix, graph):

        self.size_plant = size_plant

        self.A_plant = np.zeros((size_plant,size_plant))
        self.A_plant[0:size_plant-1,1:] = np.eye(size_plant-1)
        self.A_plant[size_plant-1:size_plant,0:] = -random_range*np.abs(np.random.rand(1, size_plant))
        self.A_plant = self.A_plant.astype(np.float64)

        self.nbr_agent = nbr_agent
        self.B_plant = B_plant
        self.tuple_output_matrix = tuple_output_matrix
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)  

        self.form = "standard" 

        self.graph = graph


class MultiAgentGroups(MultiAgentSystem):
    
    def __init__(self, nbr_agent, tuple_system_matrix, tuple_input_matrix, tuple_output_matrix, graph):

        self.tuple_system_matrix = tuple_system_matrix
        self.tuple_input_matrix = tuple_input_matrix
        self.A_plant = diag(tuple_system_matrix)
        self.size_plant = np.shape(self.A_plant)[0]
        self.nbr_agent = nbr_agent
        self.B_plant = diag(tuple_input_matrix)
        self.tuple_output_matrix = tuple_output_matrix
        self.A_plant_noisy = np.array(self.A_plant)
        self.A_plant_stabilized = np.array(self.A_plant)  

        self.form = "normal" 

        self.graph = graph

    def is_jointly_obsv(self):
        return np.linalg.matrix_rank(obsv(self.A_plant, diag(self.tuple_output_matrix))) == self.size_plant

    def obsv_index(self):
        index_array = np.zeros((self.nbr_agent,))
        for i in range(self.nbr_agent):
            index_array[i] = np.linalg.matrix_rank(obsv(self.tuple_system_matrix[i], self.tuple_output_matrix[i]))
        return index_array

# controllability/ consensus