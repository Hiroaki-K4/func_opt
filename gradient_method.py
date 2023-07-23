import numpy as np
import matplotlib.pyplot as plt


def search_local_max(f, derivative):
    x = -5
    h = 1e-5
    while np.linalg.norm(derivative(f, x, 1e-5)) > 1e-5:
        der_x = derivative(f, x, 1e-5)
        if der_x == 0:
            sgn = 0
        elif der_x > 0:
            sgn = 1
        else:
            sgn = -1
        h = sgn * np.linalg.norm(h)
        X = x
        X_h = x + h
        if f(X) < f(X_h):
            while f(X) < f(X_h):
                h = 2 * h
                X = X_h
                X_h = X + h
            x = X
            h = h / 2
        else:
            while f(X) > f(X_h):
                h = h / 2
                X_h = X_h - h
            x = X_h
            h = 2 * h

    return x


def main():
    f = lambda x:-((x-1)**2)
    derivative = lambda f, a, h:(f(a + h) - f(a - h)) / (2 * h)
    max_x = search_local_max(f, derivative)
    max_y = f(max_x)
    print("Calculate local max x value of y=-(x-1)**2")
    print("Max x: ", max_x)
    print("Max y: ", max_y)

    x = np.linspace(-5, 7, 30)
    y = -((x-1)**2)
    plt.plot(x, y)
    plt.annotate('local max', xy=(max_x, max_y),
        arrowprops=dict(facecolor='red', shrink=0.05))
    plt.show()


if __name__ == '__main__':
    main()
