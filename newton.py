import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative


def newton(f):
    x = 2
    x_a = x
    while True:
        x = x_a - derivative(f, x_a, n=1, dx=1e-7) / derivative(f, x_a, n=2, dx=1e-7)
        if abs(x - x_a) < 1.0:
            break

    return x


def main():
    x = np.linspace(-5, 7, 30)
    y = x**3 - 2 * x**2 + x + 3
    f = lambda x: x**3 - 2 * x**2 + x + 3
    ans_x = round(newton(f), 3)
    ans_y = round(f(ans_x), 3)
    print("Local min: ({0},{1})".format(ans_x, ans_y))

    plt.plot(x, y)
    plt.title("y = x ** 3 - 2 * x ** 2 + x + 3")
    label = "Local min: ({0},{1})".format(ans_x, ans_y)
    plt.annotate(
        label, xy=(ans_x, ans_y), arrowprops=dict(facecolor="red", shrink=0.05)
    )
    plt.show()


if __name__ == "__main__":
    main()
