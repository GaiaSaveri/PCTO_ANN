import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

# Download e preparazione del dataset
dataset = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Definizione del modello con ulteriori modifiche per ridurre l'overfitting
def create_model():
    model = tf.keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(10000,)),
        # Riduzione delle unità e aumento della regolarizzazione
        layers.Dropout(0.6),  # Aumento del dropout
        layers.Dense(8, activation='relu'),  # Stesse modifiche per il secondo strato
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


model = create_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello con meno epoche
history = model.fit(x_train, y_train, epochs=15, batch_size=512, validation_split=0.4,
                    verbose=2)  # Riduzione delle epoche

# Valutazione del modello
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist['accuracy']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'bo', label='Training loss')
    plt.plot(epochs, hist['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['accuracy'], 'bo', label='Training acc')
    plt.plot(epochs, hist['val_accuracy'], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(history)
