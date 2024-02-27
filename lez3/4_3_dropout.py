import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape

# Load and preprocess the dataset
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()
x_train, y_train = x_train_full[:1000], y_train_full[:1000]  # Use a subset for training
x_test, y_test = x_test_full[:200], y_test_full[:200]  # Use a subset for testing
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = x_train.reshape((-1, 28, 28, 1))  # Reshape for the CNN
x_test = x_test.reshape((-1, 28, 28, 1))


# Function to create a model with dropout
def create_model_with_dropout():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),  # Dropout layer
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),  # Dropout layer
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # Dropout layer
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to create a baseline model without dropout
def create_model_without_dropout():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the models
model_with_dropout = create_model_with_dropout()
history_with_dropout = model_with_dropout.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
                                              batch_size=64)

model_without_dropout = create_model_without_dropout()
history_without_dropout = model_without_dropout.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
                                                    batch_size=64)


# Plotting the results
def plot_results(history_with_dropout, history_without_dropout):
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].plot(history_with_dropout.history['accuracy'], label='Train with Dropout')
    axs[0].plot(history_without_dropout.history['accuracy'], label='Train without Dropout')
    axs[0].set_title('Training Accuracy')
    axs[0].legend()

    axs[1].plot(history_with_dropout.history['val_accuracy'], label='Validation with Dropout')
    axs[1].plot(history_without_dropout.history['val_accuracy'], label='Validation without Dropout')
    axs[1].set_title('Validation Accuracy')
    axs[1].legend()

    plt.show()


plot_results(history_with_dropout, history_without_dropout)
