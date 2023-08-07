import numpy as np
import random
import matplotlib.pyplot as plt



def calculate_f(x, y, z, u):
    return (u[0] * x ** 3 + y ** 3 + u[1] * x * y + u[2]) - z


def create_dataset(x, y, z, u):
    z_list = []
    for i in range(len(x)):
        noise = random.random() * 0.01
        z = calculate_f(x[i], y[i], 0, u) + noise
        z_list.append(z)

    return z_list


def gauss_newton(x, y, z, u_0):
    grad_J = 
    # H = 


def main():
    true_u = [3, -9, 27]
    u_0 = [2, 2, 2]
    x = [i for i in range(10)]
    y = [i for i in range(5, 15)]
    print(x)
    print(y)
    z = create_dataset(x, y, 0, true_u)
    print(z)
    input()
    gauss_newton(x, y, z, u_0)

    f = lambda val: 3 * val[0] ** 3 + val[1] ** 3 - 9 * val[0] * val[1] + 27
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    ax = plt.axes(projection="3d")
    ax.set_title("y = x**3 + y**3 - 9xy + 27")
    ax.contour3D(X, Y, Z, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    main()
