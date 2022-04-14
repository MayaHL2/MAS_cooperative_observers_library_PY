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

def Incidence_matrix(Adj, type_graph = "directed"):
    nbr_agent = np.shape(Adj)[0]
    edges = np.column_stack((np.array(np.where(Adj != 0)[0]), np.array(np.where(Adj != 0)[1]))) 
    B = np.zeros((nbr_agent, np.shape(edges)[0]))

    i = 0
    for edge in edges:
        if type_graph == "directed":
            B[edge[0]][i] = -1
        else: 
            B[edge[0]][i] = 1
        B[edge[1]][i] = 1
        i +=1

    return B, edges


def Incidence_column_from_edge(B, edges, edge):
    i = 0
    for e in edges:
        if np.all(e == edge): 
            return np.array(B[:,i])
        i += 1