import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape

# se disponibile, utilizziamo la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU non disponibile, verrà utilizzata la CPU.")

# Caricamento e pre-processamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# One-hot encoding delle etichette
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)

# Normalizzazione dei pixel delle immagini
x_train_norm = (x_train.astype(np.float32) / 255.0).reshape((-1, 28, 28, 1))
x_test_norm = (x_test.astype(np.float32) / 255.0).reshape((-1, 28, 28, 1))

# Definire l'architettura della CNN
model = # TODO: completare architettura

# Stampare il riepilogo del modello per mostrare l'architettura completa
model.summary()


# Compilare il modello
lr = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Addestramento del modello
n_epochs = 10
history = # TODO: fare fit del modello (come MLP)
test_loss, test_acc = # TODO: fare evaluate del modello (come MLP)

y_pred_prob = # TODO: fare predizioni usando il modello (come MLP)
y_pred = np.argmax(y_pred_prob, axis=1)

# Salvare le metriche, le etichette vere e predette
results = {
    'history': history.history,
    'test_loss': test_loss,
    'test_accuracy': test_acc,
    'true_labels': y_test.flatten(),
    'predicted_labels': y_pred,
    'predicted_probabilities': y_pred_prob,
    'true_features': x_test_norm
}

with open(f'cnn_mnist_{n_epochs}_epochs_{lr}_lr.pkl', 'wb') as file:
    pickle.dump(results, file)
