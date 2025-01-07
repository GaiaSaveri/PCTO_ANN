import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# Caricamento e pre-processamento del dataset MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

# One-hot encoding delle etichette
y_train_mnist_enc = tf.keras.utils.to_categorical(y_train_mnist)
y_test_mnist_enc = tf.keras.utils.to_categorical(y_test_mnist)

# Normalizzazione dei pixel delle immagini
x_train_mnist_norm = x_train_mnist.astype(np.float32) / 255.0
x_test_mnist_norm = x_test_mnist.astype(np.float32) / 255.0

# Creazione del modello MLP
mnist_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten per convertire l'immagine in un vettore
    tf.keras.layers.Dense(128, activation='relu'),  # Primo hidden layer
    tf.keras.layers.Dropout(0.2),  # Dropout per regolarizzazione
    tf.keras.layers.Dense(10, activation='softmax')  # Strato di output per classificazione multi-classe
])

# Compilazione del modello
lr = 0.001
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mnist_mlp.compile(
     optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
     loss=loss_fn,
     metrics=['accuracy']
)
#
# # Addestramento del modello
n_epochs = 25
mnist_mlp_history = mnist_mlp.fit(
     x_train_mnist_norm, y_train_mnist_enc,
     validation_split=0.2,
     epochs=n_epochs,
     verbose=1
)

# Plot delle perdite di training e validazione
train_loss = mnist_mlp_history.history['loss']
val_loss = mnist_mlp_history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Andamento della Loss durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Valutazione del modello sul set di test
mnist_test_loss, mnist_test_accuracy = mnist_mlp.evaluate(x_test_mnist_norm, y_test_mnist_enc, verbose=0)
print('Test accuracy: ', mnist_test_accuracy)

# Predizioni sul test set
y_pred_mnist_prob = mnist_mlp.predict(x_test_mnist_norm)
y_pred_mnist = np.argmax(y_pred_mnist_prob, axis=1)

# Matrice di confusione
conf_matrix = confusion_matrix(y_test_mnist, y_pred_mnist)
ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10)).plot(colorbar=True, cmap='viridis')
plt.title("Matrice di Confusione (MNIST)")
plt.show()

# Salvare le metriche, le etichette vere e predette
results = {
    'history': mnist_mlp_history.history,
    'test_loss': mnist_test_loss,
    'test_accuracy': mnist_test_accuracy,
    'true_labels': y_test_mnist.flatten(),
    'predicted_labels': y_pred_mnist,
    'predicted_probabilities': y_pred_mnist_prob,
    'true_features': x_test_mnist_norm
}

with open(f'mlp_mnist_{n_epochs}_epochs_{lr}_lr.pkl', 'wb') as file:
    pickle.dump(results, file)

