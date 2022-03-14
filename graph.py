import numpy as np

def random_Adjacency(nbr_agent):
    Adj = np.random.rand(nbr_agent, nbr_agent)
    Adj = np.diag(Adj)*np.eye(nbr_agent) - Adj
    return np.abs(Adj)

def Degree_matrix(Adj):
    D = np.dot(Adj,np.ones((np.shape(Adj)[0], 1)))
    D = np.reshape(D, (np.shape(D)[0],))
    return np.diag(D)

def Laplacien_matrix(A, D):
    return D - A