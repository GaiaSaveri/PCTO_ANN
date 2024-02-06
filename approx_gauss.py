import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import plot_train_val_loss


def gaussiana(x):
    return np.exp(-x*x)


n_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Disponibili: ", len(tf.config.list_physical_devices('GPU')))

x_gauss = np.linspace(-5, 5, 1000)
y_gauss = gaussiana(x_gauss)
x_train_gauss, x_test_gauss, y_train_gauss, y_test_gauss = train_test_split(x_gauss, y_gauss, test_size=0.2,
                                                                            shuffle=True)
visualizza_gaussiana = True
if visualizza_gaussiana:
    plt.plot(x_gauss, y_gauss, color='gray', label='gaussiana')
    plt.scatter(x_train_gauss, y_train_gauss, color='red', label='train points', s=5)
    plt.scatter(x_test_gauss, y_test_gauss, label='test points', color='yellow', s=5)
    plt.legend()
    plt.tight_layout()
    plt.show()

gauss_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1)
])

lr = 0.01
n_epochs = 150
gauss_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error")

train_model = True
if train_model:
    gauss_mlp_history = gauss_mlp.fit(x_train_gauss.reshape(-1, 1), y_train_gauss.reshape(-1, 1), validation_split=0.2,
                                      epochs=n_epochs, batch_size=64, verbose=1)
    gauss_mlp.save_weights('gauss_mlp_checkpoint')
    plot_train_val_loss(gauss_mlp_history)

else:
    gauss_mlp.load_weights('gauss_mlp_checkpoint')

gauss_test_loss = gauss_mlp.evaluate(x_test_gauss.reshape(-1, 1), y_test_gauss.reshape(-1, 1), verbose=0)
y_pred_gauss = gauss_mlp.predict(x_test_gauss)

plt.scatter(x_test_gauss, y_pred_gauss, label='predictions', marker='x', color='red')
plt.scatter(x_test_gauss, y_test_gauss, label='ground truth', marker='o', s=15, color='gray')
plt.legend()
plt.tight_layout()
plt.show()
