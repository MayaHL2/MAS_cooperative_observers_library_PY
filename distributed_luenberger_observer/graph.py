import numpy as np

def random_Adjacency(nbr_agent):
    Adj = np.random.rand(nbr_agent, nbr_agent)
    Adj = np.diag(Adj)*np.eye(nbr_agent) - Adj
    return np.abs(Adj)

def Degree_matrix(Adj, T = 0, type = "no pondération"):
    if type == "no pondération":
        T = np.ones((np.shape(Adj)[0], 1))
    D = np.dot(Adj, T)
    D = np.reshape(D, (np.shape(D)[0],))
    return np.diag(D)

def Laplacien_matrix(A, D):
    return D - A

def Incidence_matrix(Adj):
    nbr_agent = np.shape(Adj)[0]
    c = np.column_stack((np.array(np.where(Adj != 0)[0]), np.array(np.where(Adj != 0)[1]))) 
    B = np.zeros((nbr_agent, np.shape(c)[0]))

    i = 0
    for edge in c:
        B[edge[1]][i] = 1
        B[edge[0]][i] = -1
        i +=1

    return B
    