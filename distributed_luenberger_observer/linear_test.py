from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *

np.random.seed(3)

G = Graph(4, [[ 0, 1, 0, 1], [1, 0, 1, 0 ], [ 0, 1, 0, 1], [1, 0, 1, 0]])

print(G.Adj)

x = np.zeros(4)

# A_sys = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-2.7, -3.5, -1.4, -2.5]])
A_sys = np.array([[-2, -1, 0, 0],
                  [0, -1, 0, 0],
                  [0, -1, -1.5, 0],
                  [0, 0, 0, -2.3]])
# print(np.linalg.eig(A_sys))

# t_response = list()

std_percent = 0

# std = std_percent*np.abs(np.mean(A_sys))

B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])
C = np.eye(4)
C_sys = (np.array([[1, 0, 1, 1],[1, 0, 4, 7]]),np.array([[3, 1, 1, 1], [0, 0, -4, 0]]), np.array([[1, 0, 1, 0], [0, 1, 1, 1]]), np.array([[1, 6, 0, 2],  [0, 0, 5, 1]]))
# C_sys = (np.reshape(C[:,0], (1, -1)), np.reshape([0, 1, 0, 0], (1, -1)), np.reshape([0, 0, 1, 0], (1,-1)), np.reshape(C[:,3], (1, -1)))
MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)

print(MA.is_jointly_obsv())
print(MA.obsv_index())

# MA.step_response(7)

observer = ObserverDesign(multi_agent_system= MA, 
                            t_max= 7, 
                            x0= np.ones(16), 
                            gamma= 6, 
                            k0= np.ones(3),
                            std_noise_parameters= 0.2,
                            std_noise_sensor= 0)

# observer.feedback_control_with_observer([-1, -1.5, -2, -0.5]) 

observer.run_observer(type_observer = "output error")

# # t_response.append(observer.t_response)

# # print(t_response)
# # plt.plot(np.arange(0, len(t_response)), t_response)
# # plt.grid()
# # plt.show()

observer.plot_states()

# observer.plot_k()

observer.plot_criateria()

# observer.plot_obsv_error()


# # with open('parametersUnstable.txt', 'a') as f:
# #     f.write("std " + str(std_percent)+ "\n")
# #     f.write("k " + str(observer.k_adapt[:, int(7/0.01)-1])+ "\n")
# #     f.write("obsv error" + str(observer.obsv_error_2[:, int(7/0.01)-1])+ "\n")
# #     f.write("t_reponse "+ str(observer.t_response)+ "\n")
# #     f.write( "\n\n")