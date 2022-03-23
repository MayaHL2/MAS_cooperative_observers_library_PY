import numpy as np
from multi_observer_function import is_observable
import control as co


A = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]]
C = [0, 0, 1, 0]

print(is_observable(A, C, 4))
print(co.obsv(A,C))