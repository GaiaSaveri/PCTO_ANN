import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_train_val_loss


def plot_immagine_prob(prob, y, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    y_pred = np.argmax(prob)
    plt.xlabel('{} ({:.2f}%)'.format(int(y_pred), 100 * np.max(prob)))


def plot_prob(prob, y):
    plt.grid(False)
    plt.yticks([])
    plt.xticks(np.arange(10))
    prob_bar = plt.bar(np.arange(10), prob, color='grey')
    plt.ylim([0, 1])
    y_pred = np.argmax(prob)
    prob_bar[y].set_color('red')
    prob_bar[y_pred].set_color('green')


(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
# one-hot-encoding delle etichette categoriche
y_train_mnist_enc = tf.keras.utils.to_categorical(y_train_mnist)
y_test_mnist_enc = tf.keras.utils.to_categorical(y_test_mnist)
# normalize x features
x_train_mnist_norm = x_train_mnist.astype(np.float32)/255.0
x_test_mnist_norm = x_test_mnist.astype(np.float32)/255.0

mnist_mlp = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

lr = 0.001
n_epochs = 25
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mnist_mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn, metrics=['accuracy'])

train_model = True
if train_model:
    mnist_mlp_history = mnist_mlp.fit(x_train_mnist_norm, y_train_mnist_enc, validation_split=0.2, epochs=n_epochs,
                                      verbose=0)
    mnist_mlp.save_weights('mlp_mnist_checkpoint')
    plot_train_val_loss(mnist_mlp_history)
else:
    mnist_mlp.load_weights('mlp_mnist_checkpoint')


mnist_test_loss, mnist_test_accuracy = mnist_mlp.evaluate(x_test_mnist_norm, y_test_mnist_enc, verbose=0)
print('Test accuracy: ', mnist_test_accuracy)

y_pred_mnist_prob = mnist_mlp.predict(x_test_mnist)
y_pred_mnist = np.argmax(y_pred_mnist_prob, axis=0)

plot_idx = np.random.choice(y_test_mnist.shape[0], 1)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_immagine_prob(y_pred_mnist_prob[plot_idx], y_test_mnist[plot_idx], x_test_mnist[plot_idx].squeeze())
plt.subplot(1, 2, 2)
plot_prob(y_pred_mnist_prob[plot_idx].squeeze(), y_test_mnist[plot_idx].squeeze())
plt.show()


