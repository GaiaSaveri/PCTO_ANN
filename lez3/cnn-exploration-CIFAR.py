# Import delle librerie necessarie
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle

# Caricamento e normalizzazione del dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizzazione dei dati per migliorare l'efficienza dell'addestramento
train_images = train_images / 255.0
test_images = test_images / 255.0

# Selezione di un subset di immagini per il training e il test
num_train_subset = 10000  # Usa 10.000 immagini per l'addestramento
num_test_subset = 1000  # Usa 2.000 immagini per il test
train_images_subset = train_images[:num_train_subset]
train_labels_subset = train_labels[:num_train_subset]
test_images_subset = test_images[:num_test_subset]
test_labels_subset = test_labels[:num_test_subset]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Riduzione del numero di filtri
    layers.MaxPooling2D((2, 2)),  # Pooling per ridurre la dimensione dell'output
    layers.Flatten(),
    layers.Dense(32, activation='relu'),  # Riduzione del numero di neuroni
    layers.Dense(10, activation='softmax')  # Utilizzo di softmax per la classificazione multiclasse
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Addestramento del modello con il sottoinsieme di dati
history = model.fit(train_images_subset, train_labels_subset, epochs=10, validation_data=(test_images_subset, test_labels_subset))

# Valutazione del modello con il sottoinsieme di dati
test_loss, test_acc = model.evaluate(test_images_subset, test_labels_subset, verbose=2)
print('\nTest accuracy with subset of data:', test_acc)

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

with open('cifar_normal_results.pkl', 'wb') as file:
    pickle.dump(results, file)

# Visualizzazione delle metriche di addestramento
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Predizioni sul set di test
predicted_labels = model.predict(test_images_subset)
predicted_labels = np.argmax(predicted_labels, axis=1)

# Identificazione degli indici delle predizioni errate
wrong_indices = [i for i, (pred, true) in enumerate(zip(predicted_labels, test_labels_subset.flatten())) if pred != true]

# Selezione casuale di un sottoinsieme di indici errati per la visualizzazione
sample_wrong_indices = random.sample(wrong_indices, 9)  # Modifica questo numero per visualizzare più o meno immagini

# Nomi delle classi nel dataset CIFAR-10
class_names = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion']

# Visualizzazione delle immagini con etichette errate
plt.figure(figsize=(10, 10))
for i, wrong_index in enumerate(sample_wrong_indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images_subset[wrong_index], cmap=plt.cm.binary)
    plt.title(f"Pred: {class_names[predicted_labels[wrong_index]]}, True: {class_names[test_labels_subset[wrong_index][0]]}")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

