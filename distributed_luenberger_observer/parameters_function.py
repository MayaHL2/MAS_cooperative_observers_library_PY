from control import place, obsv
from itertools import combinations
from scipy.linalg import solve_continuous_are
from helper_function import *

def obsevability(A, C, n):
    return obsv(A, C), np.linalg.matrix_rank(obsv(A, C)), np.linalg.matrix_rank(obsv(A, C)) == n

def Mi(Ti, ki, Mid, size_unobservable):
    Mi = np.zeros((np.shape(Mid)[0] + size_unobservable, np.shape(Mid)[0] + size_unobservable))
    Mi[0:np.shape(Mid)[0],0:np.shape(Mid)[0]] = ki*Mid
    Mi[np.shape(Mid)[0]:,np.shape(Mid)[0]:] = np.eye(size_unobservable) 
    return np.dot(np.dot(Ti, Mi), np.transpose(Ti))
    
def Li(Ti, Lid, size_unobservable):
    Li = np.zeros((np.shape(Lid)[0] + size_unobservable, ))
    Li[0:np.shape(Lid)[0]] = Lid
    return np.reshape(np.dot(Ti, Li), (-1, 1))

def Lid(Aid, Hid):
    eig = -1.5*np.abs(np.random.rand(np.shape(Aid)[0])+0.75)
    return place(Aid.T,  np.reshape(Hid, (1,-1)).T, eig)

def Hid(Hi, size_obsv_space):
    return Hi[:,:size_obsv_space]

def separate_A_bar(A_bar, size_obsv_space):
    return A_bar[:size_obsv_space, :size_obsv_space], A_bar[size_obsv_space+1:, :size_obsv_space], A_bar[size_obsv_space+1:, size_obsv_space+1:]

def Mid(Aid, Lid, Hid):
    n = np.shape(Aid)[0]
    A = Aid - np.dot(Lid.T, Hid)
    return solve_continuous_are(A, np.zeros((np.shape(A)[0], 1)), np.eye(n), 1)

def new_basis(A, B, C, T):
    A_bar = np.dot(np.dot(T.T, A), T)
    B_bar = np.dot(T.T, B)
    C_bar = np.dot(C, T)
    return A_bar, B_bar, C_bar

def transformation_matrix(O, size_obs_space, n):
    
    T = np.zeros((n,n))
    norm_vector = np.linalg.norm(O, axis = 0) + 10**(-10)
    O_norm = O/norm_vector

    null_columns = np.where(
        np.all(O_norm.T == 0, axis = 1))[0]

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

    norm_vector = np.linalg.norm(O, axis = 0)

    T[:size_obs_space] = (O/norm_vector).T

    i1 = i2 = 0
    for i1 in range(n):
        if not(np.eye(n)[i1,:].tolist() in T.tolist()) and not((-np.eye(n))[i1,:].tolist() in T.tolist()):
            T[size_obs_space+i2: size_obs_space+i2+1, :] = np.eye(n)[i1,:]
            i2 += 1 

    norm = np.ones(n)
    norm[0:len(norm_vector)] = norm_vector

    T = (T.T*norm).T
    return T