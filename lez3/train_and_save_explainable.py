import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Caricamento e preparazione del dataset MNIST
(train_images, train_labels), (_, _) = datasets.mnist.load_data()
train_images, train_labels = train_images[:1000], train_labels[:1000]

# Ridimensionamento delle immagini e aggiunta di una dimensione dei canali
train_images = train_images.reshape((-1, 28, 28, 1))
train_images = train_images / 255.0  # Normalizzazione

# Definizione del modello CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Softmax per la classificazione multiclasse
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Creazione e addestramento del modello
model = create_model()
model.fit(train_images, train_labels, epochs=5)  # Addestramento per 5 epoche

# Salvataggio del modello
model.save('mnist_cnn_model.h5')
print("Modello salvato come mnist_cnn_model.h5")
