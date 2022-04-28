import numpy as np

class Graph:
    def __init__(self, nbr_agent, Adj = None, type_graph = "directed"):

        self.nbr_agent = nbr_agent

        if Adj == None:
            self.random_Adjacency()
        else:
            self.Adj = np.array(Adj)

        self.type_graph = type_graph

        self.Degree_matrix()
        self.Lap = self.Deg - self.Adj
        self.Incidence_matrix()

    def random_Adjacency(self):
        self.Adj = np.random.rand(self.nbr_agent, self.nbr_agent)
        self.Adj = np.abs(np.diag(self.Adj)*np.eye(self.nbr_agent) - self.Adj)

    def Degree_matrix(self, T = None):
        if T == None:
            T = np.ones((np.shape(self.Adj)[0], 1))
        D = np.dot(self.Adj, T)
        D = np.reshape(D, (np.shape(D)[0],))
        self.Deg = np.diag(D)

    def Incidence_matrix(self):
        nbr_agent = np.shape(self.Adj)[0]
        c = np.column_stack((np.array(np.where(self.Adj != 0)[0]), np.array(np.where(self.Adj != 0)[1]))) 
        B = np.zeros((nbr_agent, np.shape(c)[0]))

        i = 0
        for edge in c:
            if self.type_graph == "directed":
                B[edge[0]][i] = -1
            else: 
                B[edge[0]][i] = 1
            B[edge[1]][i] = 1
            i +=1

        self.Incid = B

    def access_Adjacency(self):
        return self.Adj
    
    def access_Degree(self):
        return self.Deg

    def access_Incidence(self):
        return self.Incid

    def access_Laplacien(self):
        return self.Lap
    