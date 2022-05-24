from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
from K import K

np.random.seed(3)

nbr_agent = 4

G = Graph(nbr_agent, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
# G = Graph(5, [[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 1, 0]])
print(G.Adj)

# J = [0.0820, 0.0845, 0.1377]
# m = 4.34
# d = 0.315
# lift_factor = 2*10**(-4)
# drag_factor = 7*10**(-5)

# drone = Quadrotor(m, d, J, lift_factor, drag_factor)

# A_sys, B_sys, C = drone.get_state_space()

A_sys = np.array([[1, 0, 0],
                 [0, -2, 0],
                 [0, 0, -3]])
# print(np.linalg.eig(A_sys))

# std = std_percent*np.abs(np.mean(A_sys))

B_sys = np.array([[1], [1], [1]])
# C = np.eye(4)
# C_sys = (np.array([[1, 0, 1, 1],[1, 0, 4, 7]]),np.array([[3, 1, 1, 1], [0, 0, -4, 0]]), np.array([[1, 0, 1, 0], [0, 1, 1, 1]]), np.array([[1, 6, 0, 2],  [0, 0, 5, 1]]))
# C_sys = (np.reshape(C[:,0], (1, -1)), np.reshape([0, 1, 0, 0], (1, -1)), np.reshape([0, 0, 1, 0], (1,-1)), np.reshape(C[:,3], (1, -1)))

C0 = np.array([[1, 1, 1]])
C1 = np.array([[1, 1, 1]])
C2 = np.array([[1, 1, 1]])
C3 = np.array([[1, 1, 1]])
C4 = np.array([[1, 1, 1]])

# C_faulty = drone.add_fault_to_agent("y")

# C_sys = (drone.C, drone.C_faulty, drone.C, drone.C)

# print(drone.C)
# print()
# print(drone.C_faulty)
# print()

C_sys = (C0, C1, C2, C3, C4)
# C_sys = (C4, 0)


MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)










A_plant = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [-2.7, -3.5, -1.4, -2.5]])


C1 = np.array([[1, 0, 0, 0]])
C2 = np.array([[0, 1, 0, 0]])
C3 = np.array([[0, 0, 1, 0]])
C4 = np.array([[0, 0, 0, 1]])

B_plant = np.array([[0, 0], 
                   [1, 0], 
                   [0, 0], 
                   [0, 1]])

tuple_output_matrix = (C1, C2, C3, C4)                

MA = MultiAgentSystem(A_plant, B_plant, tuple_output_matrix, G)

print("It is jointly observable:", MA.is_jointly_obsv())
print(MA.obsv_index())

# print(MA.find_faulty_agents())
# MA.get_needed_states_for_faulty_agent()


# MA.print_plant_state_space()

# print(MA.tuple_output_matrix)
# eig = MA.feedback_control([-4.7540,   -5.5407,   -3.4545,   -4.5541,   -6.4647,   -6.4815,   -2.6279,   -3.0362,   -2.2573,   -4.2040,   -2.1494, -4.2842,   -5.2457,   -3.3924,   -5.3813,   -4.9543,   -2.1199,   -4.7943,   -3.2963,   -4.0755,   -3.4176,   -5.4657, -4.2023,   -2.7843])

# K_sys = np.kron(np.eye(MA.nbr_agent), K)
# MA.feedback_control(feedback_gain= K_sys)

# MA.step_response(10, stabilized= True)

observer = ObserverDesign(multi_agent_system= MA, 
                            t_max= 10, 
                            x0= np.ones(A_plant.shape[0]), 
                            gamma= 6, 
                            k0= np.ones(4),
                            std_noise_parameters= 0,
                            std_noise_sensor= 0,
                            std_noise_relative_sensor = 0)

# observer.feedback_control_with_observer(feedback_gain= K_sys)
# observer.feedback_control_with_observer(desired_eig= -np.random.uniform(1, 6, np.shape(A_plant)[0]))

observer.run_observer(type_observer = "output error")

observer.plot_states()

observer.plot_criateria()