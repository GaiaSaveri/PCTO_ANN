import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import plot_train_val_loss


n_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Disponibili: ", len(tf.config.list_physical_devices('GPU')))

x_circles, y_circles = make_circles(n_samples=300, noise=0.1)

visualizza_circles = True
if visualizza_circles:
    # y = 0 e' rappresentato dal rosso, y = 1 e' rappresentato dal blu
    plt.scatter(x_circles[np.where(y_circles == 0)[0], 0], x_circles[np.where(y_circles == 0)[0], 1], color='red')
    plt.scatter(x_circles[np.where(y_circles == 1)[0], 0], x_circles[np.where(y_circles == 1)[0], 1], color='blue')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.show()

x_circles_train, x_circles_test, y_circles_train, y_circles_test = train_test_split(x_circles, y_circles, test_size=0.2,
                                                                                    shuffle=True)

mlp_circles = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, input_shape=(2, ), activation='relu'),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(250, activation='relu'),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')  # classificazione binaria
])

lr = 0.01
n_epochs = 500
mlp_circles.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                    metrics=['accuracy'])

train_model = True
if train_model:
    circles_mlp_history = mlp_circles.fit(x_circles_train, y_circles_train, validation_split=0.2, epochs=n_epochs,
                                          batch_size=64, verbose=1)
    mlp_circles.save_weights('mlp_circles_checkpoint')
    plot_train_val_loss(mlp_circles)
else:
    mlp_circles.load_weights('mlp_circles_checkpoint')

circles_test_loss, circles_test_accuracy = mlp_circles.evaluate(x_circles_test, y_circles_test, verbose=0)
y_pred_circles_probabilities = mlp_circles.predict(x_circles_test)
y_blu = np.where(y_pred_circles_probabilities >= 0.5)[0]
y_pred_circles = np.zeros(x_circles_test.shape[0])
y_pred_circles[y_blu] = 1
print('Test accuracy: ', circles_test_accuracy)
circles_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_circles_test, y_pred_circles),
                                                  display_labels=['rosso', 'blu'])
circles_confusion_matrix.plot(colorbar=False)
plt.show()
