import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape
import matplotlib.pyplot as plt

# Caricamento e riduzione del dataset Fashion MNIST a un subset
(x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()
x_train, y_train = x_train_full[:1000], y_train_full[:1000]  # Selezione di un subset per il training
x_test, y_test = x_test_full[:400], y_test_full[:400]  # Selezione di un subset per il test

# Preprocessing dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizzazione
x_train = x_train.reshape((-1, 28, 28, 1))  # Reshape per la CNN
x_test = x_test.reshape((-1, 28, 28, 1))

# Definizione della pipeline di data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=10,  # Reduced from 15 to 10
    zoom_range=0.05,  # Reduced from 0.1 to 0.05
    width_shift_range=0.05,  # Reduced from 0.1 to 0.05
    height_shift_range=0.05  # Reduced from 0.1 to 0.05
)


# Function to create a new model (to avoid code duplication)
def create_model():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


epochs = 30

# Train the first model with data augmentation
model_with_aug = create_model()
history_with_aug = model_with_aug.fit(
    data_augmentation.flow(x_train, y_train, batch_size=64),
    epochs=epochs,
    validation_data=(x_test, y_test),
    steps_per_epoch=x_train.shape[0] // 64
)

# Train the second model without data augmentation
model_without_aug = create_model()
history_without_aug = model_without_aug.fit(
    x_train, y_train,
    batch_size=64,
    epochs=epochs,
    validation_data=(x_test, y_test)
)


# Plotting the results
def plot_results(history_with_aug, history_without_aug):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(history_with_aug.history['accuracy'], label='Train with Augmentation')
    axs[0, 0].plot(history_without_aug.history['accuracy'], label='Train without Augmentation')
    axs[0, 0].set_title('Training Accuracy')
    axs[0, 0].legend()

    axs[0, 1].plot(history_with_aug.history['val_accuracy'], label='Val with Augmentation')
    axs[0, 1].plot(history_without_aug.history['val_accuracy'], label='Val without Augmentation')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].legend()

    axs[1, 0].plot(history_with_aug.history['loss'], label='Train with Augmentation')
    axs[1, 0].plot(history_without_aug.history['loss'], label='Train without Augmentation')
    axs[1, 0].set_title('Training Loss')
    axs[1, 0].legend()

    axs[1, 1].plot(history_with_aug.history['val_loss'], label='Val with Augmentation')
    axs[1, 1].plot(history_without_aug.history['val_loss'], label='Val without Augmentation')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].legend()

    plt.show()


# Call the plotting function
plot_results(history_with_aug, history_without_aug)
