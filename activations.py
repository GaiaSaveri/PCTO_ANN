import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# se disponibile, utilizziamo la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU non disponibile, verrà utilizzata la CPU.")

# Caricamento e pre-processamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# One-hot encoding delle etichette
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)

# Normalizzazione dei pixel delle immagini
x_train_norm = (x_train.astype(np.float32) / 255.0).reshape((-1, 28, 28, 1))
x_test_norm = (x_test.astype(np.float32) / 255.0).reshape((-1, 28, 28, 1))

# Display some of the training images
f, axes = plt.subplots(2, 10, sharey=True, figsize=(20, 4))
for i, ax in enumerate(axes.flat):
    ax.axis('off')
    ax.imshow(x_train_norm[i, :, :, 0], cmap="gray")


def create_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(24, kernel_size=(3, 3), padding='same', activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, output)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
history = model.fit(x_train_norm, y_train_enc, epochs=5, batch_size=256, validation_split=0.2)

# Predictions for visualization
predictions = model.predict(x_test[:5])

# Display some of the test images and their predicted labels
for i in range(5):
    plt.imshow(x_test_norm[i, :, :, 0], cmap="gray")
    plt.axis("off")
    predicted_label = np.argmax(predictions[i])
    plt.title(f"Predicted: {predicted_label}")
    plt.show()

# Select layers to visualize
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Choose an image to visualize
img = x_test_norm[51].reshape(1, 28, 28, 1)

# Display the selected image
plt.figure(figsize=(5, 5))
plt.imshow(img[0, :, :, 0], cmap="gray")
plt.axis('off')
plt.show()

# Get activations
activations = activation_model.predict(img)

# Layer names for the layers being visualized
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

images_per_row = 16

# Visualize the activations
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std() + 1e-5  # Avoid division by zero
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
