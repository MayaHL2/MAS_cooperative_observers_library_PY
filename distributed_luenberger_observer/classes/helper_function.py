import numpy as np

def sum_two_tuples(tuple1, tuple2):
    return (tuple1[0]+tuple2[0], tuple1[1]+ tuple2[1])

def diag(tuple_matrix): # Only works on two dimentional matrices
    shape_diag = (0,0)
    for matrix in tuple_matrix:
        shape_diag = sum_two_tuples(shape_diag, np.shape(matrix))
    matrix_diag = np.zeros(shape_diag)

    last_shape_matrix = (0,0)
    for matrix in tuple_matrix:
        matrix_diag[last_shape_matrix[0]:np.shape(matrix)[0] + last_shape_matrix[0], last_shape_matrix[1]:np.shape(matrix)[1]+last_shape_matrix[1]] = matrix
        last_shape_matrix = sum_two_tuples(np.shape(matrix), last_shape_matrix)
    return matrix_diag

def p_norm(vector, p):
    return (np.sum(np.abs(vector)**p))**(1/p)

def gaussian_noise(mu, sigma, size):
    random = np.zeros(size)
    for i in range(size[0]):
        random[i] = np.random.normal(mu[i], sigma[i])
    return random