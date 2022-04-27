from system import *
from graph import *
from luenberger_observer import *

np.random.seed(3)

G = Graph(3, [[ 0, 1,  1],[1,  0, 1 ], [ 1,   1, 0]])

A_sys = -6*np.diag(np.random.rand(4))
print(np.linalg.norm(A_sys))
B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])
C = np.eye(4)
C_sys = (C[:,0].reshape((1,-1)), np.array([0, 1, 1, 0]).reshape(1,-1), C[:,3].reshape((1,-1)))
MA = MultiAgentSystem(3, A_sys, B_sys, C_sys, G)

print(MA.is_jointly_obsv())
print(MA.obsv_index())



observer = ObserverDesign(multi_agent_system= MA, 
                            t_max= 25, 
                            x0= np.ones(4), 
                            gamma= 6, 
                            k0= np.ones(3),
                            std_noise_parameters= 0)

observer.parameters()
# observer.feedback_control_with_observer([-1, -2, -4, -5])

observer.run_observer(type_observer = "output error")
observer.plot_states()

# observer.plot_k()