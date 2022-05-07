from classes.system import *
from classes.graph import *
from classes.luenberger_observer import *

np.random.seed(3)

G = Graph(3, [[ 0, 1,  1], [1,  0, 1 ], [ 1,   1, 0]])

x = np.zeros(4)

t_max = 30
t_obs_sum = 0

while t_obs_sum < t_max:

    A_sys = np.array([[-1, 2*x[0], 0, 0], [-1, -3, 2, 0], [0, 2*x[1], -2, 6], [0, 3, x[0] + x[2], -2]])

    # A_sys = -6*np.diag(np.random.rand(4))
    print(np.linalg.norm(A_sys))
    B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])
    C = np.eye(4)
    C_sys = (C[:,0].reshape((1,-1)), np.array([0, 1, 1, 0]).reshape(1,-1), C[:,3].reshape((1,-1)))
    MA = MultiAgentSystem(3, A_sys, B_sys, C_sys, G)

    print(MA.is_jointly_obsv())
    print(MA.obsv_index())



    observer = ObserverDesign(multi_agent_system= MA, 
                                x0= x, 
                                gamma= 6, 
                                k0= np.ones(3),
                                std_noise_parameters= 0)

    observer.parameters()
    observer.feedback_control_with_observer([-1, -1.5, -2, -0.5])

    x, t_obs = observer.run_observer(type_observer = "output error")
    print(x, t_obs)
    t_obs_sum += t_obs
    observer.plot_states()

    # observer.plot_k()