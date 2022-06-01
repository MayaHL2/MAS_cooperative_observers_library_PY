from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
import time

np.random.seed(3)

nbr_agent = 2

# G = Graph(nbr_agent, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
G = Graph(nbr_agent, [[0, 1], [1, 0]])
print(G.Adj)

# J = [0.0820, 0.0845, 0.1377]
# m = 4.34
# d = 0.315
# lift_factor = 2*10**(-4)
# drag_factor = 7*10**(-5)

# drone = Quadrotor(m, d, J, lift_factor, drag_factor)

# A_sys, B_sys, C = drone.get_state_space()

# A_sys = np.array([[1, 0, 0],
#                  [0, -2, 0],
#                  [0, 0, -3]])

A_sys = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [-1, -9, -26, -24]])

# A_sys = np.array([[0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1],
#                   [-1, -9, -26, -24]])

# A_sys = np.array([[-2, -1, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, -1, -1.5, 0],
#                   [0, 0, 0, -2.3]])

# std = std_percent*np.abs(np.mean(A_sys))

# B_sys = np.array([[1], [1], [1]])
B_sys = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])

# C = np.eye(4)
# C_sys = (np.array([[1, 0, 1, 1],[1, 0, 4, 7]]),np.array([[3, 1, 1, 1], [0, 0, -4, 0]]), np.array([[1, 0, 1, 0], [0, 1, 1, 1]]), np.array([[1, 6, 0, 2],  [0, 0, 5, 1]]))
# C_sys = (np.reshape(C[:,0], (1, -1)), np.reshape([0, 1, 0, 0], (1, -1)), np.reshape([0, 0, 1, 0], (1,-1)), np.reshape(C[:,3], (1, -1)))

# C1 = np.array([[1, 1, 1]])
# C2 = np.array([[1, 1, 1]])
# C2 = np.array([[0, 0, 1]])
# C3 = np.array([[1, 1, 1]])
# C4 = np.array([[1, 0, 1]])

C1 = np.array([[0, 0, 0, 1]])
C2 = np.array([[0, 0, 0, 1]])
# C3 = np.array([[0, 0, 1, 0]])
# C4 = np.array([[0, 0, 0, 1]])

# C_faulty = drone.add_fault_to_agent("x")

# C_sys = (drone.C, drone.C_faulty, drone.C, drone.C)
# C_sys = (C1, C2, C3, C4)
C_sys = (C1, C2)
# C_sys = (C4, 0)

start = time.time()

MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)

print("It is jointly observable:", MA.is_jointly_obsv())
print("obsv index", MA.obsv_index())

# print("faulty agents", MA.find_faulty_agents())
# MA.get_needed_states_for_faulty_agent()

# print("added output", MA.added_output)
# MA.print_plant_state_space()

print(np.row_stack(MA.tuple_output_matrix))
# eig = MA.feedback_control([-4.7540,   -5.5407,   -3.4545,   -4.5541,   -6.4647,   -6.4815,   -2.6279,   -3.0362,   -2.2573,   -4.2040,   -2.1494, -4.2842,   -5.2457,   -3.3924,   -5.3813,   -4.9543,   -2.1199,   -4.7943,   -3.2963,   -4.0755,   -3.4176,   -5.4657, -4.2023,   -2.7843])

# K_sys = np.kron(np.eye(MA.nbr_agent), K)
# MA.feedback_control(feedback_gain= K_sys)
# MA.feedback_control(-np.random.uniform(1, 6, np.shape(A_sys)[0]*nbr_agent))

# MA.step_response(10)

observer = ObserverDesign(multi_agent_system= MA, 
                            t_max= 12, 
                            x0= np.ones(A_sys.shape[0]*nbr_agent), 
                            gamma= 6, 
                            k0= np.ones(nbr_agent),
                            std_noise_parameters= 0,
                            std_noise_sensor= 0,
                            std_noise_relative_sensor = 0)


# observer.feedback_control_with_observer(feedback_gain= K_sys)
# observer.feedback_control_with_observer(desired_eig= -np.random.uniform(2, 5, np.shape(A_sys)[0]*nbr_agent))

observer.run_observer(type_observer = "DFTO", lost_connexion= [[], 2, 4])

observer.plot_states()
observer.plot_criateria()
observer.plot_k()

print(time.time()- start)