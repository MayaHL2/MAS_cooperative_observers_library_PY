import numpy as np
from harold import *

def p_norm(vector, p):
    return (np.sum(np.abs(vector)**p))**(1/p)

def ctrbf(a, b, c):

    (ra,ca)= np.shape(a)
    (rb,cb)= np.shape(np.array(b, ndmin=2))

    ptjn1 = np.eye(ra)
    ajn1 = a
    bjn1 = b
    rojn1 = cb
    deltajn1 = 0
    sigmajn1 = ra
    k = np.zeros((1,ra))

    tol = ra*p_norm(a, 1)*(2**-52)

    for jj in range(ra):
        (uj,sj,vj) = np.linalg.svd(np.array(bjn1, ndmin=2))
        s = np.zeros((np.shape(uj)[0], )) 
        s[:np.shape(sj)[0]] = np.diag(sj)
        sj = s
        # print(np.shape(uj), np.shape(sj), np.shape(vj))

        (rsj,csj) = np.shape(np.array(sj, ndmin=2)) 

        p =  np.rot90(np.eye(rsj),1) 

        uj = uj*p 
        bb = uj.T*bjn1 

        roj = np.linalg.matrix_rank(bb,tol) 

        (rbb,cbb) = np.shape(np.array(bb, ndmin=2)) 

        sigmaj = rbb - roj 
        sigmajn1 = sigmaj 

        k[:,jj] = roj 

        if roj == 0: 
            break

        if sigmaj == 0:
             break

        abxy = uj.T * ajn1 * uj 

        aj   = abxy[0:sigmaj - 1,0:sigmaj - 1]
        bj   = abxy[0:sigmaj - 1,sigmaj:sigmaj+roj - 1]
        ajn1 = aj 
        bjn1 = bj 
        (ruj,cuj) = np.shape(uj) 
        ptj = ptjn1 * [uj, np.zeros((ruj,deltajn1)), np.zeros((deltajn1,cuj)), np.eye(deltajn1)]
        # print(np.shape(uj), (ruj,deltajn1), (deltajn1,cuj), (deltajn1, deltajn1)) 
        ptjn1 = ptj 
        deltaj = deltajn1 + roj 
        deltajn1 = deltaj 


    t = ptjn1.T 
    abar = t * a * t.T
    bbar = t * b 
    cbar = c * t.T

    return abar, bbar, cbar, t, k


A = np.array([[0, 1, 0, 0], [-1, 0 , 0, 0], [0, 0, 0, 2], [0, 0, -2, 0]])
C = np.reshape(np.eye(4)[3], (1,4))
B = np.reshape([0, 1, 1, 0], (4,1))


A0, B0, C0, T = staircase(A, B, C, form = "o")

print(A0)
print(B0)
print(C0)