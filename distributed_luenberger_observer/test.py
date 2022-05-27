from classes.graph import *

nbr_agent = 6
G = Graph(nbr_agent, [[0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
G.Is_strongly_connected()


G = Graph(3, [[0, 1, 0], [1, 0, 1], [0, 1, 0]]) 
faulty_agents = [0]

# print(find_minimal_connected_faulty_graph(G, [0], [0]))

new_nodes = G.find_list_minimal_connected_faulty_graph(faulty_agents)

H = G.find_sub_graph(new_nodes[0])
print(H.nbr_agent)
print(H.Adj)