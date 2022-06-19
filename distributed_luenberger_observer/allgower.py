from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *

nbr_agent = 3

G = Graph(nbr_agent, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
print(G.Adj)

A1 = np.array([[-2, 0], 
                  [0, -3]])
A2 = np.array([[-2, -6], 
                  [0, -5]])
A3 = np.array([[-6, -6], 
                  [0, -8]])

A_sys = (A1, A2, A3)

B1 = np.array([[1], [1]])
B2 = np.array([[-1], [1]])
B3 = np.array([[-1], [-1]])

B_sys = (B1, B2, B3)


C1 = [np.array([[1, 1]]), np.array([[0, 1]]), np.array([[0, 0]])]
C2 = [np.array([[1, 0]]), np.array([[1, 1]]), np.array([[0, 1]])]
C3 =  [np.array([[0, 0]]), np.array([[1, 0]]), np.array([[1, 1]])]

# C1 = [np.array([[2, 2], [2, 0]]), np.array([[0, 2], [2, 0]]), np.array([[0, 0], [0, 0]])]
# C2 = [np.array([[-1, 0], [-1, 0]]), np.array([[-1, -1], [-1, 0]]), np.array([[0, -1], [-1, 0]])]
# C3 =  [np.array([[0, 0], [0, 0]]), np.array([[1, 0], [1, 0]]), np.array([[1, 1], [1, 0]])]


C_sys = (C1, C2, C3)

MA = HeterogeneousMultiAgentGroups(A_sys, B_sys, C_sys, G)

observer = CooperativeObserverDesign(heterogeneous_MAS= MA, 
                            t_max= 10, 
                            x0= 0,
                            gamma= 6, 
                            k0= 0,
                            std_noise_parameters= 0,
                            std_noise_sensor= 0,
                            std_noise_relative_sensor = 0)

observer.run_observer()

