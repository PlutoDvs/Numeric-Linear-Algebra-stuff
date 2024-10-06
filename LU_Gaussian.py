import time
import numpy as np


def gaussian_elimination(A, b):
    n = len(A)
    M = A

    i = 0
    for x in M:
        np.append(x, b[i])
        i += 1

    for k in range(n):
        for i in range(k, n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[k], M[i] = M[i], M[k]
            else:
                pass

        for j in range(k+1, n):
            q = float(M[j][k]) / M[k][k]
            for m in range(k, n+1):
                M[j][m] -= q * M[k][m]

    x = [0 for i in range(n)]

    x[n-1] = float(M[n-1][n]) / M[n-1][n-1]
    for i in range(n-1, -1, -1):
        z = 0
        for j in range(i+1, n):
            z = z + float(M[i][j]) * x[j]
        x[i] = float(M[i][n] - z) / M[i][i]
    return x


def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i + 1):
            s1 = sum(U[k][i] * L[j][k] for k in range(j))
            U[j][i] = A[j][i] - s1
        for j in range(i, n):
            s2 = sum(U[k][i] * L[j][k] for k in range(i))
            L[j][i] = (A[j][i] - s2) / U[i][i]
    return L, U


def lu_solve(L, U, b):
    n = len(L)
    y = [0 for _ in range(n)]
    x = [0 for _ in range(n)]

    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


A = np.array([[4, -1, 0, 0, -1],
              [-1, 4, -1, 0, 0],
              [0, -1, 4, -1, 0],
              [-1, 0, 0, -1, 4]])

L, U = lu_decomposition(A)
np.random.seed(0)
b_vectors = np.random.randint(1, 101, size=(10000, 5))

start_time = time.time()
solutions = np.array([lu_solve(L, U, b) for b in b_vectors])
end_time = time.time()

total_time = end_time - start_time
print("Using LU Decomposition: ", total_time)

start_time = time.time()
solutions = np.array([gaussian_elimination(A, b) for b in b_vectors])
end_time = time.time()

total_time = end_time - start_time
print("Using Gaussian Elimination: ", total_time)

