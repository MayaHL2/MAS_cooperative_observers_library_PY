import numpy as np
import control as co
from itertools import combinations
from scipy.linalg import solve_continuous_are

def observability(A, C, n):
    return co.obsv(A, C), np.linalg.matrix_rank(co.obsv(A, C)), np.linalg.matrix_rank(co.obsv(A, C)) == n

def new_basis(A, B, C, T):
    A_bar = np.dot(np.dot(np.linalg.inv(T), A), T)
    B_bar = np.dot(np.linalg.inv(T), B)
    C_bar = np.dot(C, T)
    return A_bar, B_bar, C_bar

def transformation_matrix(O, size_obs_space, n):
    
    T = np.zeros((n,n))
    
    norm_vector = np.linalg.norm(O, axis = 0) + 10**(-10)
    O_norm = O/norm_vector

    null_columns = np.where(
        np.all(O_norm.T == 0, axis = 1))[0]

    print(null_columns)

    combi = np.arange(np.shape(O)[0])
    for c in null_columns:
        combi = np.delete(combi, np.where(combi == c)[0], axis = 0)
    combi = np.array(list(combinations(combi, 2)))

    for (i1, i2) in combi:
        if np.all(np.abs(O_norm[:, i1]-O_norm[:, i2]) < 10**(-5)) or np.all(np.abs(O_norm[:, i1]+O_norm[:, i2]) < 10**(-5)):
            O_norm[:, i2] = np.zeros(np.shape(O_norm[:, i2]))
            combi = np.delete(combi, np.where(combi == i2)[0], axis = 0)

    O = O_norm*norm_vector

    O = np.delete(O.T, np.where(
        np.all(O.T == 0, axis = 1))[0], axis=0).T

    T[:size_obs_space] = O.T

    # norm_vector = np.linalg.norm(O, axis = 0)

    # T[:size_obs_space] = (O/norm_vector).T

    # i1 = i2 = 0
    # for i1 in range(n):
    #     if not(np.eye(n)[i1,:].tolist() in T.tolist()) and not((-np.eye(n))[i1,:].tolist() in T.tolist()):
    #         T[size_obs_space+i2: size_obs_space+i2+1, :] = np.eye(n)[i1,:]
    #         i2 += 1 

    # norm = np.ones(n)
    # norm[0:len(norm_vector)] = norm_vector

    # T = (T.T*norm).T
    return T







A = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
for i in range(4):
    C = np.eye(4)[i]
    B = np.transpose(C)

    n = np.shape(A)[0]

    O, size_obs_space, is_obsv = observability(A, C, n) 
    T = transformation_matrix(O, size_obs_space, n)
    print(T)
    print("")
    A_bar, B_bar, C_bar = new_basis(A, B, C, T)

    print(size_obs_space)

    print(A)
    print(A_bar)

    print("")

    print(C)
    print(C_bar)

    print("")
    print("end")
    print("")


    