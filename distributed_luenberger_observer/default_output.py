from os import X_OK
from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
import time
from scipy.spatial.transform import Rotation as R

np.random.seed(3)

# nbr_agent = 6
# Adj = [[0, 1, 0, 0, 1, 1], 
#        [1, 0, 1, 0, 0, 0],
#        [0, 1, 0, 1, 0, 1], 
#        [0, 0, 1, 0, 1, 1], 
#        [1, 0, 0, 1, 0, 0], 
#        [1, 0, 1, 1, 0, 0]]

t_max = 10
step_UAV = 30

step = 0.01
t = np.arange(0, t_max, step)

x_ = np.array([0.4*t, 0.4*np.sin(np.pi*t), 0.6*np.cos(np.pi*t)])
p0 = np.array([np.cos(0.2*np.pi*t), np.sin(0.2*np.pi*t), np.zeros(np.max(np.shape(t),))]) + x_

angles_ = np.arccos(p0/np.sqrt(np.sum(p0**2, axis= 0)))


nbr_agent = 3
Adj = [[0, 1, 0], 
       [1, 0, 1],
       [0, 1, 0]]

G = Graph(nbr_agent, Adj)
print(G.Adj)

J = [0.0820, 0.0845, 0.1377]
m = 4.34
d = 0.315
lift_factor = 2*10**(-4)
drag_factor = 7*10**(-5)

drone = Quadrotor(m, d, J, lift_factor, drag_factor)

A_sys, B_sys, C = drone.get_state_space()
C_faulty = drone.add_fault_to_agent(type = "x")
C_faulty = drone.add_fault_to_agent(type = "y")
# C_faulty = drone.add_fault_to_agent(type = "z")
C_faulty = drone.add_fault_to_agent(type = "phi")
C_faulty = drone.add_fault_to_agent(type = "theta")
C_faulty = drone.add_fault_to_agent(type = "psy")
# C_sys = (drone.C, drone.C_faulty, drone.C, drone.C_faulty, drone.C, drone.C)
C_sys = (drone.C, drone.C_faulty, drone.C)

MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)


print("It is jointly observable:", MA.is_jointly_obsv())
print("obsv index", MA.obsv_index())

faulty_agents = MA.find_faulty_agents()
print("faulty agents", faulty_agents)

list_groups_obsv = G.find_list_minimal_connected_faulty_graph(faulty_agents)



for connected_graph_nodes in list_groups_obsv:
    start = time.time()
    graph = G.find_sub_graph(connected_graph_nodes)
    C_sys = list()
    for node in connected_graph_nodes:
        if node in faulty_agents:
            C_sys.append(drone.C_faulty)
        else:
            C_sys.append(drone.C)
    C_sys = tuple(C_sys)

    MA = MultiAgentGroups(A_sys, B_sys, C_sys, graph)


    MA.get_needed_states_for_faulty_agent()
    print("group", connected_graph_nodes)
    print("added output", MA.added_output)


    observer = ObserverDesign(multi_agent_system= MA, 
                                t_max= t_max, 
                                x0= np.zeros(A_sys.shape[0]*len(connected_graph_nodes)),
                                gamma= 6, 
                                k0= np.ones(len(connected_graph_nodes)),
                                std_noise_parameters= 0,
                                std_noise_sensor= 0,
                                std_noise_relative_sensor = 0, 
                                input= "random end")

    observer.feedback_control_with_observer(desired_eig= -np.random.uniform(0.3, 0.7, np.shape(A_sys)[0]*len(connected_graph_nodes)))

    observer.run_observer(type_observer = "output error")

    # observer.plot_states()    
    
    # defining all 3 axes
    p = (observer.x).reshape(len(connected_graph_nodes), 12, int(t_max/0.01))[0, 0:3, :]
    p_hat = (observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max/0.01))[0, 0, 0:3, :]
    print(p.shape, p_hat.shape)

    angles = (observer.x).reshape(len(connected_graph_nodes), 12, int(t_max/0.01))[0, 3:6, :]
    angles_hat = (observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max/0.01))[0, 0, 3:6, :]

    print(np.shape(p))
    distance_wings = np.max(p[0:3, :])/10
    distance_wings = 0.1
    drone.draw_quadrotor(x_, angles_, t_max, step_UAV, distance_wings)
    # drone.draw_quadrotor(p_hat, angles_hat, t_max, step_UAV, distance_wings, "blue")
    
    plt.show()
    print("execution time", time.time() - start, "s")

plt.show()