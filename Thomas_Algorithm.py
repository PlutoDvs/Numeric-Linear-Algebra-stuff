import time
import numpy as np
from scipy.linalg import solve_banded


def thomas_algorithm(l, d, u, b):
    n = len(d)

    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    # Forward elimination
    c_prime[0] = u[0] / d[0]
    d_prime[0] = b[0] / d[0]

    for i in range(1, n - 1):
        denom = d[i] - l[i - 1] * c_prime[i - 1]
        c_prime[i] = u[i] / denom
        d_prime[i] = (b[i] - l[i - 1] * d_prime[i - 1]) / denom

    d_prime[n - 1] = (b[n - 1] - l[n - 2] * d_prime[n - 2]) / (d[n - 1] - l[n - 2] * c_prime[n - 2])

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


n = 1000000
c = np.append(np.array([2]), np.array([-1 for _ in range(n-2)]))
b = np.append(np.array([5 for _ in range(n-1)]), np.array([4]))
a = np.append(np.array([1 for _ in range(n-2)]), np.array([2]))
d = np.random.randint(1, 101, size=n)

# Custom implementation
start_time = time.time()
x = thomas_algorithm(c, b, a, d)
end_time = time.time()

total_time = end_time - start_time
print("Using implemented Thomas: ", total_time)
# print(x)

# Using library
ab = np.array([np.concatenate(([0], a)), b, np.concatenate((c, [0]))])
start_time = time.time()
x = solve_banded((1, 1), ab, d)
end_time = time.time()

total_time = end_time - start_time
print("Using library: ", total_time)
# print(x)

