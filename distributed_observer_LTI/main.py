from cv2 import transpose
from graph import *
from helper_function import *
from control import place


nbr_agent = 4

step = 0.01
t_max = 20
nbr_step = int(t_max/step)

Adj = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
Degree = Degree_matrix(Adj)
Laplacien = Laplacien_matrix(Adj, Degree)
Incidence, edges = Incidence_matrix(Adj, "undirected")

A_sys = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
C = np.eye(4)
C_sys = diag((C[:,0].reshape((1,-1)), C[:,1].reshape((1,-1)), C[:,2].reshape((1,-1)), C[:,3].reshape((1,-1))))
B_sys = np.transpose([[0, 1, 0, 0], [0, 0 , 0, 1]])

eig = -np.abs(np.random.rand(np.shape(A_sys)[0])+0.75)
K_sys = place(A_sys, B_sys, eig)


A_tilde = np.kron(np.eye(nbr_agent), A_sys)
B_i = dict()
C_ii =  dict()
C_ij = dict()
for i in range(nbr_agent):
    bi = np.reshape(np.eye(nbr_agent)[:,i], (-1, 1))
    B_i[str(i)] = np.kron(bi, np.eye(np.shape(A_sys)[0]))
    C_ii[str(i)] = np.dot(C[:,i].reshape((1,-1)), np.transpose(B_i[str(i)]))
    for j in range(nbr_agent):
        if not(i==j):
            cij = Incidence_column_from_edge(Incidence, edges, [i, j])
            if cij is not None:
                C_ij[str(i) + str(j)] = np.kron(cij, np.eye(np.shape(A_sys)[0]))

print(np.shape(A_tilde), np.shape())