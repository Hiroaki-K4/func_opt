import numpy as np
import random
import matplotlib.pyplot as plt


def main():
    mu_0, sig_0 = 0, 0.5
    data_0 = np.random.normal(mu_0, sig_0, 1000)
    count, bins, ignored = plt.hist(data_0, 30, density=True)
    plt.plot(bins, 1/(sig_0 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu_0)**2 / (2 * sig_0**2) ),
            linewidth=2, color='r')

    mu_1, sig_1 = 2.5, 0.3
    data_1 = np.random.normal(mu_1, sig_1, 1000)
    count, bins, ignored = plt.hist(data_1, 30, density=True)
    plt.plot(bins, 1/(sig_1 * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu_1)**2 / (2 * sig_1**2) ),
            linewidth=2, color='r')

    mix_data = np.concatenate((data_0, data_1), axis=0)
    print(mix_data)

    w_0 = np.ones(mix_data.shape[0])
    w_1 = np.ones(mix_data.shape[0])
    half = int(mix_data.shape[0] / 2)
    # print(half)
    # input()
    # for i in range(mix_data.shape[0]):
    #     w_0[i] = random.uniform(0, 1)
    #     w_1[i] = 1 - w_0[i]

    for i in range(half):
        # w_0[i] = random.uniform(0, 1)
        # w_1[i] = 1 - w_0[i]
        w_0[i] = 0.9
        w_1[i] = 0.1

    for i in range(half, mix_data.shape[0]):
        # w_0[i] = random.uniform(0, 1)
        # w_1[i] = 1 - w_0[i]
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
        print("u_0: ", u_0)
        print("u_1: ", u_1)

        sigma_sum_0 = 0
        sigma_sum_1 = 0
        for i in range(mix_data.shape[0]):
            sigma_sum_0 += w_0[i] * (mix_data[i] - u_0) ** 2
            sigma_sum_1 += w_1[i] * (mix_data[i] - u_1) ** 2

        sigma_0 = sigma_sum_0 / N_0
        sigma_1 = sigma_sum_1 / N_1
        print("sigma_0: ", sigma_0)
        print("sigma_1: ", sigma_1)

        for i in range(mix_data.shape[0]):
            p_0 = np.exp(-(mix_data[i]-u_0)**2/2*sigma_0) / np.sqrt(2 * np.pi * sigma_0)
            p_1 = np.exp(-(mix_data[i]-u_1)**2/2*sigma_1) / np.sqrt(2 * np.pi * sigma_1)
            # print(p_0)
            # print(p_1)
            belong_prob_0 = N_0 / mix_data.shape[0] * p_0
            belong_prob_1 = N_1 / mix_data.shape[0] * p_1
            # print("belong_prob_0: ", belong_prob_0)
            # print("belong_prob_1: ", belong_prob_1)
            w_0[i] = belong_prob_0 / (belong_prob_0 + belong_prob_1)
            w_1[i] = belong_prob_1 / (belong_prob_0 + belong_prob_1)
            # print(w_0[i])
            # print(w_1[i])
            # input()

        print("w_0", w_0)
        print("old_w_0", old_w_0)
        print("w_1", w_1)
        print(np.linalg.norm(old_w_0 - w_0))
        if (np.linalg.norm(old_w_0 - w_0) / mix_data.shape[0]) < 0.001 and (np.linalg.norm(old_w_1 - w_1) / mix_data.shape[0]) < 0.001:
            break

        old_w_0 = np.copy(w_0)
        old_w_1 = np.copy(w_1)

    print("final w_0: ", w_0)
    print("final w_1: ", w_1)

    # TODO: Compare final result and init data

    plt.show()


if __name__ == "__main__":
    main()
