import random

import matplotlib.pyplot as plt
import numpy as np


def unsupervised_learning(mix_data):
    w_0 = np.ones(mix_data.shape[0])
    w_1 = np.ones(mix_data.shape[0])
    half = int(mix_data.shape[0] / 2)

    for i in range(half):
        w_0[i] = 0.9
        w_1[i] = 0.1

    for i in range(half, mix_data.shape[0]):
        w_0[i] = 0.1
        w_1[i] = 0.9

    old_w_0 = np.copy(w_0)
    old_w_1 = np.copy(w_1)
    while True:
        N_0 = np.sum(w_0)
        N_1 = np.sum(w_1)
        u_sum_0 = 0
        u_sum_1 = 0
        for i in range(mix_data.shape[0]):
            u_sum_0 += w_0[i] * mix_data[i]
            u_sum_1 += w_1[i] * mix_data[i]

        u_0 = u_sum_0 / N_0
        u_1 = u_sum_1 / N_1

        sigma_sum_0 = 0
        sigma_sum_1 = 0
        for i in range(mix_data.shape[0]):
            sigma_sum_0 += w_0[i] * (mix_data[i] - u_0) ** 2
            sigma_sum_1 += w_1[i] * (mix_data[i] - u_1) ** 2

        sigma_0 = sigma_sum_0 / N_0
        sigma_1 = sigma_sum_1 / N_1

        for i in range(mix_data.shape[0]):
            p_0 = np.exp(-((mix_data[i] - u_0) ** 2) / 2 * sigma_0) / np.sqrt(
                2 * np.pi * sigma_0
            )
            p_1 = np.exp(-((mix_data[i] - u_1) ** 2) / 2 * sigma_1) / np.sqrt(
                2 * np.pi * sigma_1
            )
            belong_prob_0 = N_0 / mix_data.shape[0] * p_0
            belong_prob_1 = N_1 / mix_data.shape[0] * p_1
            w_0[i] = belong_prob_0 / (belong_prob_0 + belong_prob_1)
            w_1[i] = belong_prob_1 / (belong_prob_0 + belong_prob_1)

        if (np.linalg.norm(old_w_0 - w_0) / mix_data.shape[0]) < 1e-3 and (
            np.linalg.norm(old_w_1 - w_1) / mix_data.shape[0]
        ) < 1e-3:
            break

        old_w_0 = np.copy(w_0)
        old_w_1 = np.copy(w_1)

    print("final w_0: ", w_0)
    print("final w_1: ", w_1)

    return w_0, w_1


def main():
    mu_0, sig_0 = 0, 0.5
    data_0 = np.random.normal(mu_0, sig_0, 1000)
    mu_1, sig_1 = 2.5, 0.3
    data_1 = np.random.normal(mu_1, sig_1, 1000)

    fig = plt.figure(figsize=(16, 9))
    fig_graph_0 = fig.add_subplot(121)
    fig_graph_1 = fig.add_subplot(122)
    fig_graph_0.hist(data_0, 30, density=True)
    fig_graph_0.hist(data_1, 30, density=True)
    fig_graph_0.set_title(label="Input data")

    mix_data = np.concatenate((data_0, data_1), axis=0)
    w_0, w_1 = unsupervised_learning(mix_data)

    class_0 = []
    class_1 = []
    for i in range(mix_data.shape[0]):
        if w_0[i] >= w_1[i]:
            class_0.append(mix_data[i])
        else:
            class_1.append(mix_data[i])

    fig_graph_1.hist(class_0, 30, density=True)
    fig_graph_1.hist(class_1, 30, density=True)
    fig_graph_1.set_title(
        label="The result of classification from unsupervised learning"
    )

    plt.show()


if __name__ == "__main__":
    main()
