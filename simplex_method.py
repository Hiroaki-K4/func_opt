import sys

import numpy as np


def transform_mat(target_mat, target_row, target_col):
    target_mat[target_row] = (
        target_mat[target_row] / -target_mat[target_row, target_col]
    )
    for i in range(target_mat.shape[0]):
        if i == target_row:
            continue
        ratio = target_mat[i, target_col]
        for j in range(target_mat.shape[1]):
            if j == 0:
                continue
            elif j == target_col:
                target_mat[i, j] = -target_mat[target_row, 0] * ratio
            else:
                target_mat[i, j] += target_mat[target_row, j] * ratio

    swap_num = -target_mat[target_row, 0]
    target_mat[target_row, 0] = -target_mat[target_row, target_col]
    target_mat[target_row, target_col] = swap_num

    return target_mat


def check_convergence(target_mat):
    for i in range(2, target_mat.shape[1]):
        if target_mat[target_mat.shape[0] - 1, i] > 0:
            return False

    return True


def simplex_method(target_mat):
    print("target_mat:")
    print(target_mat)
    target_cols = []
    while True:
        if check_convergence(target_mat):
            break
        max_inc_rate = 0
        for i in range(2, target_mat.shape[1]):
            if i in target_cols:
                continue
            x_min = sys.maxsize
            for j in range(target_mat.shape[0] - 1):
                x = -target_mat[j, 1] / target_mat[j, i]
                if x < x_min:
                    x_min = x
                    tmp_target_row = j
                    tmp_target_col = i

            inc_rate = target_mat[target_mat.shape[0] - 1, i] * x_min
            if inc_rate > max_inc_rate:
                max_inc_rate = inc_rate
                target_row = tmp_target_row
                target_col = tmp_target_col

        target_cols.append(target_col)
        target_mat = transform_mat(target_mat, target_row, target_col)

    max_value = target_mat[target_mat.shape[0] - 1, 1]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Max value is ", max_value)


if __name__ == "__main__":
    target_mat = np.array(
        [[1, 60, -2, -8], [1, 60, -4, -4], [0, 0, 29, 45]], dtype=np.float64
    )
    simplex_method(target_mat)
