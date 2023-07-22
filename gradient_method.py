import numpy as np

def derivative(f,a,h):
    return (f(a + h) - f(a - h)) / (2 * h)


def search_local_max(f):
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
    max_x = search_local_max(f)
    print("Max x: ", max_x)


if __name__ == '__main__':
    main()
