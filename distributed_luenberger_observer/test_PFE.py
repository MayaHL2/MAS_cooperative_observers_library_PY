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


type = ["DLO", "DFTO"]

for j in range(2):
    print(type[j])
    t_response = list([])
    static_error = list([])
    for k in range(100):
        print(k)
        if j == 0:
            g = 6
        else:
            g = -0.02
        observer = ObserverDesign(multi_agent_system= MA, 
                                    t_max= None, 
                                    x0= np.ones(A_sys.shape[0]),
                                    x_hat_0= [[-1, 2], [0.5, -1.3]],
                                    gamma= g, 
                                    k0= np.ones(nbr_agent),
                                    std_noise_parameters= k/100,
                                    std_noise_sensor= 0,
                                    std_noise_relative_sensor = 0)

        observer.run_observer(type_observer = type[j])

        if observer.t_max == None:
            break
        t_response = t_response + [observer.t_max]
        # static_error = static_error + [observer.static_error]

    print(t_response)
    plt.plot(np.arange(0,np.size(t_response)), t_response)    
    plt.xlabel("added parametric noise/parameters' values")
    plt.ylabel("settling time")

plt.grid()
plt.show()