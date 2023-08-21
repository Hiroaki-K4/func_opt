import numpy as np
import sys


def transform_mat(target_mat, target_row, target_col):
    target_mat[target_row] = target_mat[target_row] / -target_mat[target_row, target_col]
    print(target_mat)
    input()
    for i in range(target_mat.shape[0]):
        if i == target_row:
            continue

    return target_mat


def simplex_method(target_mat):
    print(target_mat)
    max_inc_rate = 0
    for i in range(2, target_mat.shape[1]):
        x_min = sys.maxsize
        for j in range(target_mat.shape[0] - 1):
            x = -target_mat[j, 1] / target_mat[j, i]
            if x < x_min:
                x_min = x
                tmp_target_row = j
                tmp_target_col = i

        inc_rate = target_mat[target_mat.shape[0]-1, i] * x_min
        if inc_rate > max_inc_rate:
            max_inc_rate = inc_rate
            target_row = tmp_target_row
            target_col = tmp_target_col

    target_mat = transform_mat(target_mat, target_row, target_col)

    print(max_inc_rate)
    print(target_row)
    print(target_col)
    print(target_mat)


if __name__ == "__main__":
    target_mat = np.array([[1, 60, -2, -8], [1, 60, -4, -4], [0, 0, 29, 45]], dtype=np.float64)
    simplex_method(target_mat)
