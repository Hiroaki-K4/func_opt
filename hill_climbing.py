import numpy as np
import matplotlib.pyplot as plt
from sympy import *


def search_local_max(f, derivative, fixed_x_y, calculate_grad_f):
    x = 1
    h = 1e-5
    old_grad_f = calculate_grad_f(fixed_x_y)
    new_pos = fixed_x_y + x * old_grad_f
    while np.linalg.norm(derivative(calculate_grad_f(new_pos), old_grad_f)) > 1e-5:
        der_x = derivative(calculate_grad_f(new_pos), old_grad_f)
        if der_x == 0:
            sgn = 0
        elif der_x > 0:
            sgn = 1
        else:
            sgn = -1
        h = sgn * np.linalg.norm(h)
        X = x
        X_h = x + h
        if f(fixed_x_y + X * calculate_grad_f(fixed_x_y)) < f(
            fixed_x_y + X_h * calculate_grad_f(fixed_x_y)
        ):
            while f(fixed_x_y + X * calculate_grad_f(fixed_x_y)) < f(
                fixed_x_y + X_h * calculate_grad_f(fixed_x_y)
            ):
                h = 2 * h
                X = X_h
                X_h = X + h
            x = X
            h = h / 2
        else:
            while f(fixed_x_y + X * calculate_grad_f(fixed_x_y)) > f(
                fixed_x_y + X_h * calculate_grad_f(fixed_x_y)
            ):
                h = h / 2
                X_h = X_h - h
            x = X_h
            h = 2 * h

        # Update grad and new position
        old_grad_f = calculate_grad_f(new_pos)
        new_pos = fixed_x_y + x * calculate_grad_f(fixed_x_y)

    return x


def hill_climbing(f):
    x = Symbol("x")
    y = Symbol("y")
    z = -(x**2) + (-(y**2))
    grad_x = diff(z, x)
    grad_y = diff(z, y)
    calculate_grad_f = lambda val: np.array(
        [float(grad_x.subs([(x, val[0])])), float(grad_y.subs([(y, val[1])]))]
    )
    F_derivative = lambda grad_f, old_grad_f: np.dot(grad_f, old_grad_f)

    x_y = np.array([-6, -6])
    x_move = np.array([1000000, 1000000])
    while np.linalg.norm(x_move) > 1e-2:
        t = search_local_max(f, F_derivative, x_y, calculate_grad_f)
        x_move = np.dot(t, calculate_grad_f(x_y))
        x_y = x_y + x_move

    return x_y


def main():
    f = lambda val: -((val[0]) ** 2) + (-(val[1] ** 2))
    local_max_pos = hill_climbing(f)
    round_local_max_pos = np.array(
        [round(local_max_pos[0], 3), round(local_max_pos[1], 3)]
    )
    print("Local max: ({0},{1})".format(round_local_max_pos[0], round_local_max_pos[1]))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))
    ax = plt.axes(projection="3d")
    ax.set_title("y = -x ** 2 - y ** 2")
    ax.contour3D(X, Y, Z, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    label = "Local max: ({0},{1})".format(
        round_local_max_pos[0], round_local_max_pos[1]
    )
    ax.text(round_local_max_pos[0], round_local_max_pos[1], 0, label, None)
    plt.show()


if __name__ == "__main__":
    main()
