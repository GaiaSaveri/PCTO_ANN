import numpy as np
import matplotlib.pyplot as plt


# funzioni di attivazione (non linearità) utilizzate all'interno delle rete neurali
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
   return np.maximum(0, x)


def softmax(list_x):
    exp_x = np.exp(list_x)
    somma_exp_x = np.sum(np.exp(list_x))
    return exp_x / somma_exp_x


x_list = np.arange(-10, 10)

fig, axs = plt.subplots(1, 3, figsize=(12, 5))
axs[0].plot(x_list, np.array([relu(x) for x in x_list]))
axs[0].set_title('ReLU')
axs[1].plot(x_list, np.array([sigmoid(x) for x in x_list]))
axs[1].set_title('Sigmoid')
axs[2].plot(x_list, np.array(softmax(x_list)))
axs[2].set_title('Softmax')
plt.tight_layout()
plt.show()
