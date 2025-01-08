import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Preprocessing dei dati
x_train_norm = (x_train / 255.0).reshape((-1, 28, 28, 1))
x_test_norm = (x_test / 255.0).reshape((-1, 28, 28, 1))

# One-hot encoding delle etichette
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)


# Function to create a model with dropout
def create_model_with_dropout():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        # TODO: completare con i layer suggeriti
        # per tutte le convoluzioni usiamo activation='relu' e padding='same'
        # conv: 32 filtri, di kernel_size=3

        # batch normalization

        # conv: 32 filtri, di kernel_size=3

        # batch normalization

        # max pooling: pool_size=2

        # dropout: 0.25

        # conv: 64 filtri, di kernel_size=3

        # batch normalization

        # conv: 64 filtri, di kernel_size=3

        # batch normalization

        # max pooling: pool_size=2

        # dropout: 0.25

        # flatten

        # dense: 128 neuroni, activation='relu'

        # batch normalization

        # dropout: 0.5

        # dense: 10 neuroni, activation='softmax'

    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# Function to create a baseline model without dropout
def create_model_without_dropout():
    model = Sequential([
        Reshape((28, 28, 1), input_shape=(28, 28, 1)),
        # TODO: identica a sopra, ma senza dropout
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# Train the models
model_with_dropout = create_model_with_dropout()
history_with_dropout = model_with_dropout.fit(x_train_norm, y_train_enc, epochs=20, validation_split=0.2, batch_size=64)

model_without_dropout = create_model_without_dropout()
history_without_dropout = model_without_dropout.fit(x_train_norm, y_train_enc, epochs=20, validation_split=0.2, batch_size=64)


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