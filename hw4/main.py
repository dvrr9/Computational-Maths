import numpy as np

'''
(py')' - qy = -f(x)
p = e^x
q = 0
f(x) = -e^(x)
'''

def p(x):
    return np.exp(x)


def q(x):
    return 0


def f(x):
    return -np.exp(x)


def y(x):
    return (-np.e * x + x - np.exp(1 - x) + np.e) / (1 - np.e)


def finite_element_method(N):
    h = 1 / N
    x = np.linspace(0, 1, N + 1)

    b = np.array([0] + [
        -1 / (h ** 2) * (np.exp(x[j]) - np.exp(x[j - 1])) for j in range(1, N + 1) # integrate(-p(x) + q(x)(x - x_{j-1})(x_j - x))dx
    ]) 
    phi_scalar = np.array([0] + [
        1 / (h ** 2) * (np.exp(x[j + 1]) - np.exp(x[j - 1])) for j in range(1, N)
    ])

    A = np.zeros((N + 1, N + 1))
    A[0, 0] = 1
    A[N, N] = 1
    for j in range(1, N):
        A[j, j - 1] = b[j]
        A[j, j] = phi_scalar[j]
        A[j, j + 1] = b[j + 1]

    def phi_integral(x, k):
        return np.exp(x) * (k - x + 1) / h
    
    target = np.array([0] + [
        phi_integral(x[j], x[j - 1]) - phi_integral(x[j - 1], x[j - 1]) - phi_integral(x[j + 1], x[j + 1]) + phi_integral(x[j], x[j + 1])
        for j in range(1, N) # integrate(f(x)phi_j(x))dx
    ] + [0])

    y = np.linalg.solve(A, target)
    return y


def estimate_error(N_values, K):
    eps = 1e-7
    real_values = [y(x) for x in np.linspace(0, 1, K + 1)]
    h_square = [1 / (x ** 2) for x in N_values]
    max_error = []

    for N in N_values:
        fem_values = finite_element_method(N)
        approx_values = np.zeros(K + 1)
        for i in range(K + 1):
            x = i / K
            ind = int(x * N + eps)
            if ind != N:
                approx_values[i] = fem_values[ind] + (fem_values[ind + 1] - fem_values[ind]) * (x - ind / N) * N
            else:
                approx_values[i] = fem_values[N]
        max_error.append(max(abs(real_values - approx_values)))
    
    return h_square, max_error


if __name__ == '__main__':
    N_values = np.arange(10, 1000, 10)
    h_square, max_error = estimate_error(N_values, 100000)
