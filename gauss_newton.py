import random

import matplotlib.pyplot as plt
import numpy as np


def calculate_f(x, y, z, u):
    return (u[0] * x**3 + u[1] * y**2 + u[2] * x * y + 27) - z


def df_du_0(x):
    return x**3


def df_du_1(y):
    return y**2


def df_du_2(x, y):
    return x * y


def create_dataset(x, y, z, u):
    z_list = []
    for i in range(len(x)):
        noise = random.random() * 0.01
        z = calculate_f(x[i], y[i], 0, u) + noise
        z_list.append(z)

    return z_list


def calculate_nabla_u(x, y):
    return np.array([[df_du_0(x)], [df_du_1(y)], [df_du_2(x, y)]])


def gauss_newton(x, y, z, u_0):
    u = u_0
    while True:
        grad_J_sum = np.zeros([3, 1])
        H_sum = np.zeros([3, 3])
        for i in range(len(x)):
            nabla_u = calculate_nabla_u(x[i], y[i])
            grad_J_sum += calculate_f(x[i], y[i], z[i], u) * nabla_u
            H_sum += np.dot(nabla_u, nabla_u.T)

        u_move = np.dot(np.linalg.inv(H_sum), grad_J_sum)
        u = u - u_move
        if np.linalg.norm(u_move) < 1e-8:
            break

    return u


def main():
    true_u = np.array([3, 2, -9])
    u_0 = np.array([1.9, 1.9, 4.6]).reshape(3, 1)
    x = [i for i in range(10)]
    y = [i for i in range(-5, 5)]
    z = create_dataset(x, y, 0, true_u)
    predict_u = gauss_newton(x, y, z, u_0)
    print("True U: ", true_u)
    print("Predicted U: ", predict_u[:, 0])

    f = lambda val: 3 * val[0] ** 3 + 2 * val[1] ** 2 - 9 * val[0] * val[1] + 27
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    ax = plt.axes(projection="3d")
    ax.set_title("y = 3x**3 + 2y**2 - 9xy + 27")
    ax.contour3D(X, Y, Z, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    main()
