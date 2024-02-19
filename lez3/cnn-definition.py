# Importare le librerie necessarie
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Caricare il dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizzare i dati delle immagini
train_images, test_images = train_images / 255.0, test_images / 255.0

# Ridimensionare le immagini per la CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Definire l'architettura della CNN
model = models.Sequential([
    # Aggiungere il primo strato convoluzionale
    # 32: numero di filtri (kernels) che la rete apprenderà
    # (3, 3): dimensione di ciascun filtro (3x3)
    # activation='relu': funzione di attivazione ReLU (Rectified Linear Unit)
    # input_shape=(28, 28, 1): dimensione delle immagini in input (28x28 pixel, 1 canale di colore poiché le immagini sono in scala di grigi)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Aggiungere uno strato di pooling
    # (2, 2): fattore di pooling per ridurre le dimensioni spaziali (dividendo altezza e larghezza per 2)
    layers.MaxPooling2D((2, 2)),

    # Appiattire l'output dallo strato di pooling per passarlo agli strati densamente connessi
    layers.Flatten(),

    # Aggiungere uno strato densamente connesso (fully connected)
    # 64: numero di neuroni nello strato
    # activation='relu': funzione di attivazione ReLU
    layers.Dense(64, activation='relu'),

    # Aggiungere lo strato di output
    # 10: numero di neuroni nello strato di output, uno per ciascuna delle 10 classi di cifre (0-9)
    layers.Dense(10)
])

# Stampare il riepilogo del modello per mostrare l'architettura completa
model.summary()


# Compilare il modello
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Esegui questo script per vedere l'architettura della CNN e preparati per l'addestramento nel prossimo script.
