import numpy as np
from functools import reduce

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

        if np.all(Adj) == None:
            self.random_Adjacency()
        else:
            self.Adj = np.array(Adj)

        self.type_graph = type_graph

        self.Degree_matrix()
        self.Lap = self.Deg - self.Adj
        self.Incidence_matrix()

        self.Is_strongly_connected()

        if not(self.strongly_connected):
            print("This graph is not strongly connected")
        else: 
            print("This graph is strongly connected")
            
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
    
    
    def Is_strongly_connected(self, visited = 0, reachable = 0, i = 0):
        """ This is a recursive function to test if the graph is 
        strongly connected
        Arguments:
            visited: all the already visited nodes.
            reachable: all the nodes that we know arereachable.
            i: counter.
        Returns:
            True if it is strongly connected False otherwise.
        """
        if i == 0:
            visited = np.array([False] * self.nbr_agent)
            reachable = np.array(visited)
            reachable[0] = True

        visited[i] = True
        reachable[np.where(self.Adj[i,:]>0)[0]] = True


        if np.all(reachable):
            self.strongly_connected = True
            return True
        else:
            k = np.intersect1d(np.where(self.Adj[i,:]>0)[0], np.where(visited == False)[0])
            if len(k) == 0:
                self.strongly_connected = False
                return False 
            else:
                return self.Is_strongly_connected(visited, reachable, k[0])


    def find_minimal_connected_faulty_graph(self, faulty_nodes, faulty_graph_connected):
        """ This is a recursive function to find for a given node from the
        faulty nodes, the other nodes that it needs to be connected to in 
        order to obtain a minimal connected graph for which the system 
        would be jointly observable. 
        Arguments:
            faulty_nodes: the agents with faults.
            reachable: initialy, it would be the faulty agent for which we 
            are trying to find its neighboring observable graph. After 
            recursion, it is the list of the faulty agents that will be in
            the same connected minimal graph with observable agents.
        Returns:
            The minimal connected graph of a given agent for which the 
            system would be jointly observable.
        """
        node_neighbors = np.where(self.Adj[faulty_graph_connected[len(faulty_graph_connected)-1]]> 0)[0]
        faulty_neighbors = np.intersect1d(faulty_nodes, node_neighbors)
        non_faulty_neighbors = np.intersect1d(np.setdiff1d(np.arange(self.nbr_agent), faulty_nodes), node_neighbors)

        if len(non_faulty_neighbors) != 0:
            return np.append(faulty_graph_connected, non_faulty_neighbors[0])
        else:
            return self.find_minimal_connected_faulty_graph(faulty_nodes, np.append(faulty_graph_connected,faulty_neighbors[0]))

    def find_list_minimal_connected_faulty_graph(self, faulty_agents):
        """ Using the function find_minimal_connected_faulty_graph(),
        find all the minimal connected graph for each faulty agent
        and append the remaining observable agents.
        Arguments:
            faulty_agents: the agents with faults.
        Returns:
            The list of minimal connected graphs on which the 
            distributed observer can be performed.
        """
        list_groups_obsv = list([])
        for agent in faulty_agents:
            graph_agent = self.find_minimal_connected_faulty_graph(faulty_agents, [agent])
            exists = False
            for previous_graph in list_groups_obsv:
                if np.all(np.isin(graph_agent, previous_graph)):
                    exists = True

            if not exists:
                list_groups_obsv.append(graph_agent)

        # Add to the list the remaining agents splited as they are an observable
        # group on their own

        # PROBLEM when faulty<->nonfaulty<->faulty
        if not(np.all(np.isin(np.arange(self.nbr_agent), list_groups_obsv))):
            list_groups_obsv = list_groups_obsv + np.split(np.setdiff1d(np.arange(self.nbr_agent),
                                                    reduce(np.union1d, (list_groups_obsv))), 
                            len(np.setdiff1d(np.arange(self.nbr_agent), 
                                            reduce(np.union1d, (list_groups_obsv))))
                                        )
                            
        return list_groups_obsv

    def find_sub_graph(self, new_nodes):
        """ This function returns a subgraph of self.
        Arguments:
            new_nodes: the nodes contained in the subgraph.
        Returns:
            The subgraph.
        """
        # new_nodes = np.zeros(self.nbr_agent)
        # for k in faulty:
        #     new_nodes += self.Adj[k]
        # new_nodes = np.sort(np.append(np.where(new_nodes>0)[0], faulty))

        other_nodes = np.setdiff1d(np.arange(self.nbr_agent), new_nodes)
        
        new_adj = np.delete(self.Adj, other_nodes, axis = 0)
        new_adj = np.delete(new_adj, other_nodes, axis = 1)

        # return len(new_nodes), new_adj
        return Graph(len(new_nodes), new_adj)

