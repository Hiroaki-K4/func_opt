import numpy as np
import matplotlib.pyplot as plt


def hill_climbing(f):
    x = 0


def main():
    f = lambda x,y:-(x**2)+(-(y**2))
    derivative = lambda f, a, h:(f(a + h) - f(a - h)) / (2 * h)
    hill_climbing(f)

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()


if __name__ == '__main__':
    main()
