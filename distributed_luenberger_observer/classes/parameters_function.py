from control import place, ctrb
from scipy.linalg import solve_continuous_are
from .helper_function import *

def Mi(Ti, ki, Mid, size_unobservable):
    Mi = np.zeros((np.shape(Mid)[0] + size_unobservable, np.shape(Mid)[0] + size_unobservable))
    Mi[0:np.shape(Mid)[0],0:np.shape(Mid)[0]] = ki*Mid
    Mi[np.shape(Mid)[0]:,np.shape(Mid)[0]:] = np.eye(size_unobservable) 
    return np.dot(np.dot(Ti, Mi), np.transpose(Ti))
    
def Li(Ti, Lid, size_unobservable):
    Li = np.zeros((np.shape(Lid)[0] + size_unobservable, np.shape(Lid)[1]))
    Li[0:np.shape(Lid)[0], :] = Lid
    return np.reshape(np.dot(Ti, Li), (-1, np.shape(Lid)[1]))

def Lid(Aid, Hid, minEig = 2, maxEig = 4):
    eig = -np.random.uniform(minEig, maxEig, np.shape(Aid)[0])
    print("eig observer", eig)
    return place(Aid.T,  np.reshape(Hid, (-1,np.shape(Aid)[0])).T, eig)

def Hid(Hi, size_obsv_space):
    return Hi[:,:size_obsv_space]

def separate_A_bar(A_bar, size_obsv_space):
    # Aid, Air, Aiu
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