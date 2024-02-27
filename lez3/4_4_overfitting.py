# 1. Import Required Libraries
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

# 2. Download and Prepare the Dataset
dataset = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000)


# Function to vectorize the sequences
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Prepare the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# 3. Define the Model
def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# 4. Compile the Model
model = create_model()  # Change to False to see the effect of no regularization
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model
history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.4, verbose=2)

# 6. Evaluate the Model
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")


# 7. Plotting Utility
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
