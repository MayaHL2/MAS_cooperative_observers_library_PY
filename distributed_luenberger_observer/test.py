import numpy as np
from control import obsv
from harold import staircase
from classes.quadrotor import *
from classes.helper_function import *


A_sys = np.array([[-1, 0, 0],
                 [0, -2, 0],
                 [0, 0, -3]])
print(np.linalg.eig(A_sys))

B_sys = np.array([[1], [1], [1]])

C1 = np.array([[0, 0, 1]])
C2 = np.array([[1, 1, 1]])
C3 = np.array([[1, 1, 1]])
C4 = np.array([[1, 1, 1]])

print(np.linalg.matrix_rank(obsv(A_sys, C1)))

A_bar, B_bar, C_bar, T = staircase(A_sys, B_sys, C1, form = "o")

print(A_bar)
print(T)