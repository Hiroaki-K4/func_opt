import numpy as np
import matplotlib.pyplot as plt
from sympy import *


def newton_multi_var(grad_F, H):
    x = np.array([3, 2])
    while True:
        H_inv = np.linalg.inv(np.float64(H(x)))
        x_move = np.dot(H_inv, np.float64(grad_F(x)))
        x = x - x_move
        if np.linalg.norm(x_move) < 1e-8:
            break

    return x


def main():
    x, y = symbols("x y")
    f = x**3 + y**3 - 9 * x * y + 27
    grad_x = f.diff(x)
    grad_y = f.diff(y)
    calculate_grad_f = lambda val: np.array(
        [
            grad_x.subs([(x, val[0]), (y, val[1])]),
            grad_y.subs([(x, val[0]), (y, val[1])]),
        ]
    )
    H = lambda val: np.array(
        [
            [
                f.diff(x, x).subs([(x, val[0]), (y, val[1])]),
                f.diff(x, y).subs([(x, val[0]), (y, val[1])]),
            ],
            [
                f.diff(x, y).subs([(x, val[0]), (y, val[1])]),
                f.diff(y, y).subs([(x, val[0]), (y, val[1])]),
            ],
        ]
    )
    local_min = newton_multi_var(calculate_grad_f, H)
    round_local_min_pos = np.array([round(local_min[0], 3), round(local_min[1], 3)])
    print("Local min: ({0},{1})".format(round_local_min_pos[0], round_local_min_pos[1]))

    f = lambda val: val[0] ** 3 + val[1] ** 3 - 9 * val[0] * val[1] + 27
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
    ax.scatter([local_min[0]], [local_min[1]], [f(local_min)], color="red")
    label = "Local min: ({0},{1})".format(
        round_local_min_pos[0], round_local_min_pos[1]
    )
    ax.text(round_local_min_pos[0], round_local_min_pos[1], f(local_min), label, None)
    plt.show()


if __name__ == "__main__":
    main()
