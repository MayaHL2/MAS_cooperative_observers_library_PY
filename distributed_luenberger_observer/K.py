import numpy as np

f = open("K.txt", "r")
K = np.zeros((8, 24))

i = 0
for line in f:
    j = 0
    for n in line.split():
        K[i, j] = float(n)
        j += 1
    i += 1
f.close()


f = open("H.txt", "r")
H = np.zeros((24, 24))

i = 0
for line in f:
    j = 0
    for n in line.split():
        H[i, j] = float(n)
        j += 1
    i += 1
f.close()