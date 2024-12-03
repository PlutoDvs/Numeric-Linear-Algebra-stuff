import time
from scipy.linalg import solve_banded
import numpy as np


def jacobi(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = A.shape[0]
    x = np.zeros_like(b) if x0 is None else x0
    x_new = np.zeros_like(x)

    for k in range(max_iterations):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1

        x = x_new.copy()

    raise ValueError("Jacobi method did not converge within the maximum number of iterations")


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iterations=1000):
    n = A.shape[0]
    x = np.zeros_like(b) if x0 is None else x0

    for k in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1

        x = x_new

    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations")


def sor(A, b, omega=1.0, x0=None, tol=1e-10, max_iterations=1000):
    n = A.shape[0]
    x = np.zeros_like(b, dtype=float) if x0 is None else x0

    for k in range(max_iterations):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i, n))
            x_new[i] = x[i] + (omega / A[i, i]) * (b[i] - s1 - s2)

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1

        x = x_new

    raise ValueError("SOR method did not converge within the maximum number of iterations")


def thomas_algorithm(a, b, c, f):
    n = len(b)

    alpha = np.zeros(n)
    beta = np.zeros(n - 1)
    y = np.zeros(n)

    alpha[0] = b[0]

    for i in range(1, n):
        beta[i - 1] = a[i - 1] / alpha[i - 1]
        alpha[i] = b[i] - beta[i - 1] * c[i - 1]

    y[0] = f[0]
    for i in range(1, n):
        y[i] = f[i] - beta[i - 1] * y[i - 1]

    x = np.zeros(n)
    x[n - 1] = y[n - 1] / alpha[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / alpha[i]

    return x


A = np.zeros((100, 100), dtype=float)
np.fill_diagonal(A, 1 + 2 * 0.25)
np.fill_diagonal(A[1:], -0.25)
np.fill_diagonal(A[:, 1:], -0.25)

b = np.ones(100, dtype=float)

# Jacobi
start_time = time.time()
_, iterations = jacobi(A, b, tol=5e-13)
end_time = time.time()
# print("Jacobi Solution:", solution)
print("Jacobi Iterations:", iterations, ", Time:", end_time - start_time)

# Gauss_Seidel
start_time = time.time()
_, iterations = gauss_seidel(A, b, tol=5e-13)
end_time = time.time()
# print("Gauss-Seidel Solution:", solution)
print("Gauss-Seidel Iterations:", iterations, ", Time:", end_time - start_time)

# SOR
Omega = [1.1, 1.3, 1.5, 1.9]
for omega in Omega:
    start_time = time.time()
    _, iterations = sor(A, b, omega=omega, tol=5e-13)
    end_time = time.time()
    print("SOR Iterations with omega =", omega, ":", iterations, ", Time:", end_time - start_time)


# Thomas
u = np.array([-0.25 for _ in range(99)])
d = np.array([1 + 2 * 0.25 for _ in range(100)])
l = np.array([-0.25 for _ in range(99)])

start_time = time.time()
x = thomas_algorithm(u, d, l, b)
end_time = time.time()
print("Using Thomas: ", end_time - start_time)

# Using library
ab = np.array([np.concatenate(([0], l)), d, np.concatenate((u, [0]))])
start_time = time.time()
x = solve_banded((1, 1), ab, b)
end_time = time.time()
print("Using library: ", end_time - start_time)

