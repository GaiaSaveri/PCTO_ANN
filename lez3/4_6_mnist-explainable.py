import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, train_labels = train_images[:10000], train_labels[:10000]

# Reshape the images to add the channel dimension (grayscale)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Normalize the pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Display some of the training images
f, axes = plt.subplots(2, 10, sharey=True, figsize=(20, 4))
for i, ax in enumerate(axes.flat):
    ax.axis('off')
    ax.imshow(train_images[i, :, :, 0], cmap="gray")


def create_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(24, kernel_size=(3, 3), padding='same', activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(10, activation="softmax")(x)

    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.20, random_state=42)

model = create_model()
history = model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test))

# Predictions for visualization
predictions = model.predict(test_images[:5])

# Display some of the test images and their predicted labels
for i in range(5):
    plt.imshow(test_images[i, :, :, 0], cmap="gray")
    plt.axis("off")
    predicted_label = np.argmax(predictions[i])
    plt.title(f"Predicted: {predicted_label}")
    plt.show()

# Select layers to visualize
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Choose an image to visualize
img = test_images[51].reshape(1, 28, 28, 1)

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
