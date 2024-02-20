# Importazione delle librerie necessarie
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Caricamento e normalizzazione del dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Ridimensionamento delle immagini per adattarle all'input della CNN
train_images = train_images.reshape((60000, 28, 28, 1))  # 60000
test_images = test_images.reshape((10000, 28, 28, 1))  # 10000

# Definizione dell'architettura della CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classi per i numeri da 0 a 9
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Seleziona un sottoinsieme di immagini per l'addestramento
num_train_images = 6000
train_images_subset = train_images[:num_train_images]
train_labels_subset = train_labels[:num_train_images]

# Seleziona un sottoinsieme di immagini per il test
num_test_images = 1000
test_images_subset = test_images[:num_test_images]
test_labels_subset = test_labels[:num_test_images]

# Addestramento del modello con il sottoinsieme di dati
history = model.fit(train_images_subset, train_labels_subset, epochs=10, validation_data=(test_images_subset, test_labels_subset))

# Valutazione del modello con il sottoinsieme di dati
test_loss, test_acc = model.evaluate(test_images_subset, test_labels_subset, verbose=2)
print('\nTest accuracy with subset of data:', test_acc)

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

# Generazione delle predizioni sul set di test
predictions = model.predict(test_images)

# Definizione del numero di righe e colonne per la visualizzazione delle immagini
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols


# Funzione per visualizzare immagini e predizioni
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i].reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         true_label),
               color=color)


# Visualizzazione delle predizioni
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
plt.tight_layout()
plt.show()
