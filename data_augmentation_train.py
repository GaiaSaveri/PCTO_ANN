import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape

# Caricamento e riduzione del dataset Fashion MNIST a un subset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocessing dei dati
x_train_norm = (x_train / 255.0).reshape((-1, 28, 28, 1))
x_test_norm = (x_test / 255.0).reshape((-1, 28, 28, 1))

# One-hot encoding delle etichette
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)

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
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


epochs = 30

# Train with data augmentation
model_with_aug = create_model()
history_with_aug = model_with_aug.fit(
    data_augmentation.flow(x_train_norm, y_train_enc, batch_size=64),  # includiamo la data augmentation nel training
    epochs=epochs,
    validation_data=(x_test_norm, y_test_enc),
    steps_per_epoch=x_train_norm.shape[0] // 64
)