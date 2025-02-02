import torch
import matplotlib.pyplot as plt
import random


def create_data(w, b, data_num):
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b
    noise = torch.normal(0, 0.001, y.shape)

    y += noise
    return x, y


num = 500
true_w = torch.tensor([8.1, 2, 2, 4])
true_b = torch.tensor(1.1)
X, Y = create_data(true_w, true_b, num)

plt.scatter(X[:, 3], Y, 1)
plt.show()

