# Importare le librerie necessarie
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle

# PROVARE A MODIFICARE QUESTO SCRIPT IN MODO CHE MIGLIORINO LE PERFORMANCES

# Caricare e preparare il dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Selezione di un subset di immagini per il training e il test
num_train_subset = 10000
num_test_subset = 1000
train_images_subset = train_images[:num_train_subset]
train_labels_subset = train_labels[:num_train_subset]
test_images_subset = test_images[:num_test_subset]
test_labels_subset = test_labels[:num_test_subset]


# Definizione della funzione per creare il modello
def create_optimized_model():
    model = models.Sequential([
        # Primo blocco convoluzionale
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        # Secondo blocco convoluzionale
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Appiattimento e strati densamente connessi
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Uso softmax per la classificazione multiclasse
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# Creare e addestrare il modello ottimizzato
model = create_optimized_model()
history = model.fit(train_images_subset, train_labels_subset, epochs=10,
                    validation_data=(test_images_subset, test_labels_subset))

# Valutazione del modello
test_loss, test_acc = model.evaluate(test_images_subset, test_labels_subset, verbose=2)
print('\nTest accuracy with subset:', test_acc)

# Utilizzare il modello per fare previsioni sul subset di test
predictions = model.predict(test_images_subset)
predicted_labels = np.argmax(predictions, axis=1)

# Salvare le metriche, le etichette vere e predette
results = {
    'history': history.history,
    'test_loss': test_loss,
    'test_accuracy': test_acc,
    'true_labels': test_labels_subset.flatten(),
    'predicted_labels': predicted_labels
}

with open('cifar_optimized_results.pkl', 'wb') as file:
    pickle.dump(results, file)

# Visualizzazione delle metriche di addestramento e validazione
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
