from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *

np.random.seed(3)

nbr_agent = 4

G = Graph(nbr_agent, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

print(G.Adj)

A_sys = np.array([[-1, 0],
                 [0, -2]])
# print(np.linalg.eig(A_sys))

std_percent = 0

# std = std_percent*np.abs(np.mean(A_sys))

B_sys = [[1], [1]]
# C = np.eye(4)
# C_sys = (np.array([[1, 0, 1, 1],[1, 0, 4, 7]]),np.array([[3, 1, 1, 1], [0, 0, -4, 0]]), np.array([[1, 0, 1, 0], [0, 1, 1, 1]]), np.array([[1, 6, 0, 2],  [0, 0, 5, 1]]))
# C_sys = (np.reshape(C[:,0], (1, -1)), np.reshape([0, 1, 0, 0], (1, -1)), np.reshape([0, 0, 1, 0], (1,-1)), np.reshape(C[:,3], (1, -1)))

C1 = np.array([[1, 0]])
C2 = np.array([[1, 0]])
C3 = np.array([[1, 0]])
C4 = np.array([[1, 0]])
C_sys = (C1, C2, C3, C4)

MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)

print("It is jointly observable:", MA.is_jointly_obsv())
print(MA.obsv_index())

MA.print_plant_state_space()

# print(MA.find_faulty_agents())
# MA.get_needed_states_for_faulty_agent()

# eig = MA.feedback_control(feedback_gain = K)
# eig = MA.feedback_control([-4.7540,   -5.5407,   -3.4545,   -4.5541,   -6.4647,   -6.4815,   -2.6279,   -3.0362,   -2.2573,   -4.2040,   -2.1494, -4.2842,   -5.2457,   -3.3924,   -5.3813,   -4.9543,   -2.1199,   -4.7943,   -3.2963,   -4.0755,   -3.4176,   -5.4657, -4.2023,   -2.7843])

# MA.step_response(7, stabilized= True)

observer = ObserverDesign(multi_agent_system= MA, 
                            t_max= 7, 
                            x0= np.ones(A_sys.shape[0]*nbr_agent), 
                            gamma= 6, 
                            k0= np.ones(4),
                            std_noise_parameters= 0,
                            std_noise_sensor= 0,
                            std_noise_relative_sensor = 0.4)

# observer.feedback_control_with_observer(feedback_gain = K) 

observer.run_observer(type_observer = "output error")

# # # t_response.append(observer.t_response)

# # print(t_response)
# # plt.plot(np.arange(0, len(t_response)), t_response)
# # plt.grid()
# # plt.show()

observer.plot_states()

MA.print_plant_state_space()

# # observer.plot_k()

# observer.plot_criateria()

# # observer.plot_obsv_error()


# with open('parametersUnstable.txt', 'a') as f:
#     f.write("std " + str(std_percent)+ "\n")
#     f.write("k " + str(observer.k_adapt[:, int(7/0.01)-1])+ "\n")
#     f.write("obsv error" + str(observer.obsv_error_2[:, int(7/0.01)-1])+ "\n")
#     f.write("t_reponse "+ str(observer.t_response)+ "\n")
#     f.write( "\n\n")