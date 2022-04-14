import matplotlib.pyplot as plt
from graph import *
from control import place
from parameters_function import *
from harold import staircase

np.random.seed(1)

nbr_agent = 4

step = 0.01
t_max = 20
nbr_step = int(t_max/step)

Adj = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
Degree = Degree_matrix(Adj)
Laplacien = Laplacien_matrix(Adj, Degree)

# A_sys = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
# A_sys = np.array([[37, -1.5, 0, 0], [0, 19 , 0, 0], [0, 7.1, -49, 0], [0, 0, 0, -2]])
A_sys = np.array([[-2, -1, 0, 0], [0, -5 , 0, 0], [0, -1, -7, 0], [0, 0, 0, -23]])

# size_agent = 4
# A_sys = np.zeros((size_agent,size_agent))
# A_sys[0:size_agent-1,1:] = np.eye(size_agent-1)
# A_sys[size_agent-1:size_agent,0:] = -5*np.abs(np.random.rand(1, size_agent))

print(A_sys)

print("eig", np.linalg.eig(A_sys)[0])

C = np.eye(4)
C_sys = diag((C[:,0].reshape((1,-1)), C[:,1].reshape((1,-1)), C[:,2].reshape((1,-1)), C[:,3].reshape((1,-1))))
B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])


# eig = -1*np.abs(np.random.rand(np.shape(A_sys)[0])+0.5)
# eig = [-10, -15, -7.5, -8]
# print(eig)
# K_sys = place(A_sys, B_sys, eig)
K_sys = np.zeros((np.shape(B_sys)[1], np.shape(A_sys)[0]))

T = dict()
A_bar = dict()
B_bar = dict()
C_bar = dict()

Ad = dict()
Hd = dict()
Md = dict()
Ld = dict()
L = dict()
M = dict()


k = np.ones(nbr_agent)

gamma = 6

for i in range(nbr_agent):
    O, size_obsv, observable = obsevability(A_sys, C[:, i], nbr_agent)
    # Should I put O.T or O in the transformation matrix
    # T[str(i)] = transformation_matrix(O.T, size_obsv, nbr_agent)
    A_bar[str(i)], B_bar[str(i)], C_bar[str(i)], T[str(i)] = staircase(A_sys, B_sys, np.reshape(C[:,i], (1,np.shape(A_sys)[0])), form = "o")
    # A_bar[str(i)], B_bar[str(i)], C_bar[str(i)] = new_basis(A_sys, B_sys, C[:,i], T[str(i)])
    Ad[str(i)], _, _ = separate_A_bar(A_bar[str(i)], size_obsv)
    Hd[str(i)] = Hid(C_bar[str(i)], size_obsv)
    Ld[str(i)] = Lid(Ad[str(i)], Hd[str(i)])
    Md[str(i)] = Mid(Ad[str(i)], Ld[str(i)], Hd[str(i)])
    M[str(i)] = Mi(T[str(i)], k[i], Md[str(i)], np.shape(A_sys)[0] - size_obsv)
    L[str(i)] = Li(T[str(i)], np.reshape(Ld[str(i)], (-1,)).T,  np.shape(A_sys)[0] - size_obsv)

    # print("numéro", i)
    # print("T", T[str(i)])
    # print("L", Ld[str(i)])
    # print("M", Md[str(i)])
    # print()

    # print("numéro", i)
    # print("T", T[str(i)])
    # print("A", A_bar[str(i)])
    # print("B", B_bar[str(i)])
    # print("C", C_bar[str(i)])
    # print()


M = [v for v in M.values()]
M = diag(M)

L = [v for v in L.values()]
L = diag(L)

x = np.zeros((np.shape(A_sys)[0], nbr_step))
x[:, 0] = np.transpose([1, 0.5, 1, 0])
x_hat = 5*np.random.rand(nbr_agent, np.shape(A_sys)[0], nbr_step)
x_hat_concatenated = np.reshape(x_hat, (nbr_agent*np.shape(A_sys)[0], nbr_step))


y = np.zeros((np.shape(C_sys)[0], nbr_step))
y_hat = np.zeros((nbr_agent, np.shape(C_sys)[0], nbr_step))

A_sys_concatenated = np.kron(np.eye(nbr_agent), A_sys)
# A_sys_concatenated_noisy = np.random.normal(np.mean(A_sys_concatenated), 0.01, np.shape(A_sys_concatenated))
B_sys_concatenated = np.kron(np.eye(nbr_agent), B_sys)
K_sys_concatenated = np.kron(np.eye(nbr_agent), K_sys)

# u_sys = np.cos(np.arange(0, t_max, step))
u_sys = np.zeros((2, nbr_step))
u_concatenated = np.reshape(np.array([u_sys for _ in range(nbr_agent)]), (np.shape(B_sys_concatenated)[1], -1))


Laplacien_m = np.kron(Laplacien, np.eye(np.shape(A_sys)[0]))

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(A_sys, x[ :,i]) - step*np.dot(np.dot(B_sys, K_sys), x[ :,i]) + step*np.reshape(np.dot(B_sys, u_sys[:,i]), (-1,)) + x[:,i] 
    x_concatenated = np.array([x[:, i] for _ in range(nbr_agent)])
    x_concatenated = np.reshape(x_concatenated, (np.shape(x_concatenated)[0]*np.shape(x_concatenated)[0], ))

    
    # y[:,i+1] = np.dot(C_sys, x[:,i+1])
    # y_concatenated[:,i+1] = np.reshape(y_hat[:,:,i+1] , (nbr_agent*nbr_agent,))

    x_hat_concatenated[:,i+1] = step*np.dot(A_sys_concatenated - np.dot(B_sys_concatenated, K_sys_concatenated), x_hat_concatenated[:,i]) + step*np.reshape(np.dot(B_sys_concatenated, u_concatenated[:,i]), (-1, )) + x_hat_concatenated[:,i]  + step*np.dot(np.dot(L, C_sys), x_concatenated - x_hat_concatenated[:,i]) + step*gamma*np.dot(np.dot(np.linalg.inv(M), -Laplacien_m), x_hat_concatenated[:, i])
    x_hat[:,:, i+1] = np.reshape(x_hat_concatenated[:,i+1], (nbr_agent, np.shape(A_sys)[0],))

for j in range(x.shape[0]):
    # plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]))
    # plt.plot(np.arange(0,t_max, step), np.transpose(x_hat[:, j,:]))
    plt.plot(np.arange(0,t_max, step), np.transpose(x[j,:]-x_hat[:,j,:])) 

plt.grid()
plt.show()

