from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *
from classes.quadrotor import *
import time


np.random.seed(3)

t_max = [25, 150, 150]
# t_max = [2.50, 2.50, 2.50]
step_UAV = 50

step = 0.01

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

# xd1 = drone.helix_helicoidal_trajectory(12, 100, 100, 0.1, t_max, 0.01)
# xd2 = drone.helix_helicoidal_trajectory(19, 50, 70, 0.1, t_max, 0.01)

x_hat_0 = np.row_stack((np.column_stack(([-2, -3, -2.5]*np.ones((1,3)), np.pi/10*np.zeros((1,3)), np.zeros((1,6)), [2.8, -3.0, -1.5]*np.ones((1,3)), np.pi/2*np.zeros((1,3)), np.zeros((1,6)))),
                        np.column_stack(([-2, -3, -2.5]*np.ones((1,3)), np.pi/10*np.zeros((1,3)), np.zeros((1,6)), [2.8, 3.0, -1.5]*np.ones((1,3)), np.pi/2*np.zeros((1,3)), np.zeros((1,6))))))

A_sys, B_sys, C = drone.get_state_space()
C_faulty = drone.add_fault_to_agent(type = "x")
C_faulty = drone.add_fault_to_agent(type = "y")
# C_faulty = drone.add_fault_to_agent(type = "phi")
C_faulty = drone.add_fault_to_agent(type = "theta")
C_faulty = drone.add_fault_to_agent(type = "psy")
# C_faulty = drone.add_fault_to_agent(type = "dx")
# C_faulty = drone.add_fault_to_agent(type = "dy")
# C_faulty = drone.add_fault_to_agent(type = "dz")
C_sys = (drone.C, drone.C_faulty, drone.C)

MA = MultiAgentGroups(A_sys, B_sys, C_sys, G)


print("It is jointly observable:", MA.is_jointly_obsv())
print("obsv index", MA.obsv_index())

faulty_agents = MA.find_faulty_agents()
print("faulty agents", faulty_agents)

list_groups_obsv = G.find_list_minimal_connected_faulty_graph(faulty_agents)

# listh = [0.05, 0.1, 0.2]

listh = [0]

# for connected_graph_nodes in list_groups_obsv:
for b in range(len(listh)):
    # for k in range(len(list_groups_obsv)):
    k = 1
    connected_graph_nodes = list_groups_obsv[k]
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

    # print()
    # for k in range(len(connected_graph_nodes)):
    #     print()
    #     print(MA.tuple_output_matrix[k].shape)
    #     print(MA.tuple_output_matrix[k])
    #     print("NEWWWWW")
    #     print()

    if len(connected_graph_nodes) == 1:
        # desired_x = (xd1)
        x_hat_0 = np.column_stack(([-1, 3, 0.5]*np.ones((1,3)), np.pi/20*np.zeros((1,3)), np.zeros((1,6))))
        x0 = np.zeros(A_sys.shape[0]*len(connected_graph_nodes))
    else:
        # desired_x = (xd1, xd2)
        x0 = np.row_stack((np.zeros((A_sys.shape[0], 1)), np.expand_dims([1.5, -3, -3]*np.ones((3, )), axis = 1), np.zeros((9, 1))))

    print(x0.T)
    for x in MA.tuple_output_matrix :
        for y in x:
            if len(np.where(y <0)[0]) > 0:
                print(np.where(y <0)[0])

    observer = ObserverDesign(multi_agent_system= MA, 
                                t_max= t_max[b], 
                                x0= x0,
                                x_hat_0= x_hat_0,
                                desired_states= None, 
                                gamma= 0.1, 
                                k0= 0.1*np.ones(len(connected_graph_nodes)),
                                std_noise_parameters= listh[b],
                                std_noise_sensor= 0,
                                std_noise_relative_sensor = 0, 
                                input= "zero")

    eig = -np.random.uniform(0.1, 0.2, np.shape(A_sys)[0]*len(connected_graph_nodes))
    print("Controller eigenvalue", eig)
    # observer.feedback_control_with_observer(desired_eig= eig)

    observer.run_observer(type_observer = "DLO", tol_t_response= 10**(-2))

    # observer.plot_states()    
    # observer.plot_states(saveFile= "image/agent2/")  
    # observer.plot_obsv_error()
    # observer.plot_obsv_error(saveFile= "image/agent2/")
    # observer.plot_criateria()
    # observer.plot_k()

    # ax = plt.axes(projection ='3d')
    for j in range(len(connected_graph_nodes)):
        # ax = plt.axes(projection ='3d')
        p = (observer.x).reshape(len(connected_graph_nodes), 12, int(t_max[b]/0.01))[j, 0:3, :]
        p_hat = np.mean((observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max[b]/0.01))[:, j, 0:3, :], axis= 0)

        angles = (observer.x).reshape(len(connected_graph_nodes), 12, int(t_max[b]/0.01))[0, 3:6, :]
        angles = np.arctan2(np.sin(angles), np.cos(angles))*180/np.pi
        # angles[angles< 0] = angles[angles< 0] + 2*np.pi
        angles_hat = np.mean((observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max[b]/0.01))[:, j, 3:6, :], axis= 0)
        angles_hat = np.arctan2(np.sin(angles_hat), np.cos(angles_hat))*180/np.pi
        # angles_hat[angles_hat< 0] = angles_hat[angles_hat< 0] + 2*np.pi

        v = (observer.x).reshape(len(connected_graph_nodes), 12, int(t_max[b]/0.01))[j, 6:9, :]
        v_hat = np.mean((observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max[b]/0.01))[:, j, 6:9, :], axis= 0)

        w = 180/np.pi*(observer.x).reshape(len(connected_graph_nodes), 12, int(t_max[b]/0.01))[j, 9:12, :]
        w_hat = 180/np.pi*np.mean((observer.x_hat).reshape(-1, len(connected_graph_nodes), 12, int(t_max[b]/0.01))[:, j, 9:12, :], axis= 0)

        real = (p, angles, v, w)
        estimate = (p_hat, angles_hat, v_hat, w_hat)

        obsv_error2 = (observer.obsv_error_2[:3, :], observer.obsv_error_2[3:6, :], observer.obsv_error_2[6:9, :], observer.obsv_error_2[9:12, :])

        label = ("Euclidean position (m)", "Euler angles (deg)", "velocity (m/s)", "angular velocity (deg/s)")
        # label = ("Integral square observation error of the Euclidean position", "Integral square observation error of the Euler angles", "Integral square observation error of the velocities", "Integral square observation error of the angular velocities")
        color = ["m", "#FFA500", "#ff6961", "#77DD77", "#5CA0FF", "#FFF35A", "#762EFF", "#5AECFF", "#00E02D", "#B10FFF"]

        # plt.plot(np.arange(0, t_max[b], 0.01), np.transpose(np.dot(observer.K_sys, np.mean(observer.x_hat, axis =0))[0]))
        # plt.grid()
        # plt.xlabel("time")
        # plt.ylabel("The controller U_thrust")
        # print()
        fig, ax = plt.subplots(nrows=2, ncols=2)

        fig.set_figwidth(19.20)
        fig.set_figheight(10.80)

        s = 0
        for row in ax:
            for col in row:
                col.plot(np.arange(0, t_max[b], 0.01), np.transpose(real[s]), c ="#1E7DF0", ls = "dashed")
                for l in range(3):
                    col.plot(np.arange(0, t_max[b], 0.01), np.transpose(estimate[s][l,:]), color = color[l])
                    # col.plot(np.arange(0, t_max[b], 0.01), np.transpose(obsv_error2[s][l,:]), color = color[l])
                    # col.plot(np.arange(0, t_max[b], 0.01), np.transpose(estimate[s][l,:] - real[s]), color = color[l])
                col.set_title(label[s])
                col.grid()
                s += 1 

        # distance_wings = 0.1
        # drone.draw_quadrotor(ax, p_hat, angles_hat, t_max[b], step_UAV, distance_wings, "blue", "#5F5F5F")
        # drone.draw_quadrotor(ax, p, angles, t_max[b], step_UAV, distance_wings, "green")
        
        # plt.savefig("image/quadrotor" + str(j+2) + "parameters" + str(int(listh[b]*100)) + ".png", bbox_inches='tight', dpi=1000)

        plt.show()

    print("execution time", time.time() - start, "s")



# J'AI CHANGE LES VALEURS DES PARAMETRES DE L_I de 1/ 2 à 2/4 POUR 
# ASSURER LA CONVERENCE DE L'OBSV EN CAS DE NOISE PARAMETRIC 