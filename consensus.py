import numpy as np
import matplotlib.pyplot as plt
from graph import *

nbr_agent = 10
size_agent = 10

step = 0.01
t_max = 10
nbr_step = int(t_max/step)

Adj = random_Adjacency(nbr_agent)
D = Degree_matrix(Adj)
L = Laplacien_matrix(Adj, D)

x = 5*np.random.rand(size_agent*nbr_agent, nbr_step)

for i in range(nbr_step-1):
    x[:,i+1] = step*np.dot(-np.kron(L,np.eye(size_agent)),x[:,i]) + x[:,i] 

for j in range(size_agent*nbr_agent):
    plt.plot(np.arange(0,t_max, step), x[j,:])

plt.grid()
plt.show()