import numpy as np

class Graph:
    """ This class defines a graph.
    """
    def __init__(self, nbr_agent, Adj = None, type_graph = "directed"):
        """ 
        Arguments:
            nbr_agent: the number of agents connected through the 
            graph.
            Adj: the adjacency matrix of the graph. If None, it is
            defined randomly.
            type_graph: directed or undirected.
        Returns:
            None
        """

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
        """ This function creates a random adjacency matrix with
        nbr_agent lines and columns.
        Arguments:
            None
        Returns:
            None
        """
        self.Adj = np.random.rand(self.nbr_agent, self.nbr_agent)
        self.Adj = np.abs(np.diag(self.Adj)*np.eye(self.nbr_agent) - self.Adj)

    def Degree_matrix(self, T = None):
        """ This function defines the degree matrix of the graph
        Arguments:
            T: the weighting array for the degree matrix. if None, 
            T = [1, 1, ..., 1]
        Returns:
            None
        """
        if T == None:
            T = np.ones((np.shape(self.Adj)[0], 1))
        D = np.dot(self.Adj, T)
        D = np.reshape(D, (np.shape(D)[0],))
        self.Deg = np.diag(D)

    def Incidence_matrix(self):
        """ This function defines the incidence matrix of the graph
        Arguments:
            None
        Returns:
            None
        """
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

    