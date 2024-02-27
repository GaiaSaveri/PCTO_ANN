import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import datasets

# Caricamento del modello addestrato
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Caricamento del dataset MNIST
(_, _), (test_images, test_labels) = datasets.mnist.load_data()

# Trova un indice di un'immagine '7' nel test set
index_of_7 = next(i for i, label in enumerate(test_labels) if label == 9)

# Prepara l'immagine '7' per il modello
img = test_images[index_of_7].reshape(1, 28, 28, 1) / 255.0

# Visualizza l'immagine '7'
plt.imshow(test_images[index_of_7], cmap='gray')
plt.title("Immagine Originale: 7")
plt.axis('off')
plt.show()


# Funzione per visualizzare le feature map di un certo strato
def visualize_layer(layer_name):
    # Crea un modello intermedio per ottenere le output del layer specificato
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_prediction = intermediate_model.predict(img)

    # Visualizza le prime 8 feature map del layer
    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(intermediate_prediction[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.suptitle(f"Feature Map dello strato: {layer_name}")
    plt.show()


# Visualizza le feature map per i primi due strati convoluzionali
visualize_layer('conv2d')  # Assumi che il nome del primo strato convoluzionale sia 'conv2d'
visualize_layer('conv2d_1')  # Nome del secondo strato convoluzionale
visualize_layer('conv2d_2')