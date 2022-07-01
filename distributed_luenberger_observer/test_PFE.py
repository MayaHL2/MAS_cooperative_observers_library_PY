from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
import time

np.random.seed(3)

nbr_agent = 2

G = Graph(nbr_agent, [[0, 1], [1, 0]])
print(G.Adj)

A_sys = np.array([[-2, 0], 
                  [0, -3]])

B_sys = np.array([[1], [1]])

C1 = np.array([[1, 0]])
C2 = np.array([[0, 1]])

C_sys = (C1, C2)

start = time.time()

MA = MultiAgentSystem(A_sys, B_sys, C_sys, G)

print("It is jointly observable:", MA.is_jointly_obsv())
print("obsv index", MA.obsv_index())

# MA.step_response(t_max = 7, x0= [-1, 0, 1, 2])

t_response = list([])
static_error = list([])

for k in range(1):

    observer = ObserverDesign(multi_agent_system= MA, 
                                t_max= 3, 
                                x0= np.ones(A_sys.shape[0]),
                                x_hat_0= [[-1, 2], [0.5, -1.3]],
                                gamma= 6, 
                                k0= np.ones(nbr_agent),
                                std_noise_parameters= 0,
                                std_noise_sensor= 0,
                                std_noise_relative_sensor = 0)

    # observer.feedback_control_with_observer([-1, -2])
    # observer.run_observer(type_observer = "DLO", lost_connexion= [[0], 2, 4], tol_t_response= 10**(-2))
    observer.run_observer(type_observer = "DFTO")
    # observer.run_observer(type_observer = "DFTO")
    # if observer.t_max == None:
    #     break
    # t_response = t_response + [observer.t_max]
    # static_error = static_error + [observer.static_error]

observer.plot_states(saveFile = "image/plant/stable/DFTO/")
observer.plot_criateria(saveFile = "image/plant/stable/DFTO/")
observer.plot_k(saveFile = "image/plant/stable/DFTO/")
observer.plot_obsv_error(saveFile = "image/plant/stable/DFTO/")

# observer.plot_states(saveFile = "image/plant/stable/DLO/")
# observer.plot_criateria(saveFile = "image/plant/stable/DLO/")
# observer.plot_k(saveFile = "image/plant/stable/DLO/")
# observer.plot_obsv_error(saveFile = "image/plant/stable/DLO/")

observer.plot_states()
observer.plot_criateria()
observer.plot_k()
observer.plot_obsv_error()

# print(time.time()- start)

# print(t_response)
# print(static_error)

# plt.plot(np.arange(0,np.size(t_response)), t_response)    
# # plt.legend([""], loc ="lower right")
# plt.xlabel("added parametric noise/parameters' values")
# plt.ylabel("settling time")
# # plt.title("The response time to a step versus parametric noise")
# plt.grid()
# plt.show()


# plt.plot(np.arange(0,np.shape(static_error)[0]), static_error)    

# # plt.title("Static error versus parametric noise")
# plt.xlabel("added parametric noise/parameters' values")
# plt.ylabel("steady-state error")
# plt.grid()
# plt.show()