import numpy as np


def check_orthogonality(Q):
    n = Q.shape[1]
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            dot_product = np.dot(Q[:, i], Q[:, j])
            result.append(((i, j), dot_product))
    return result


def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))

    for i in range(n):
        vi = A[:, i]
        projection_sum = np.zeros(m)
        for j in range(i):
            projection = np.dot(Q[:, j], vi) * Q[:, j]
            projection_sum += projection

        orthogonal_vector = vi - projection_sum

        Q[:, i] = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    return Q


def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))

    for i in range(n):
        q = A[:, i]
        for j in range(i):
            q = q - np.dot(Q[:, j], q) * Q[:, j]

        Q[:, i] = q / np.linalg.norm(q)

    return Q


A = np.array([[1, 1, 1],
              [1e-8, 0, 0],
              [0, 1e-8, 0],
              [0, 0, 1e-8]], dtype=float)

Q1 = modified_gram_schmidt(A)
Q2 = gram_schmidt(A)
print("Original algorithm: Q:")
print(Q2)
print("Modified algorithm: Q:")
print(Q1)
print(check_orthogonality(Q2))
print(check_orthogonality(Q1))
