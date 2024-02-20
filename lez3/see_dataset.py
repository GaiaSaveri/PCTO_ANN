import tensorflow as tf
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(mnist_train_images, mnist_train_labels), _ = tf.keras.datasets.mnist.load_data()

# Carica il dataset CIFAR-10
(cifar_train_images, cifar_train_labels), _ = tf.keras.datasets.cifar10.load_data()


# Funzione per visualizzare le immagini di un dataset
def display_images(images, labels, title, cmap=None):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # MNIST images are grayscale, CIFAR-10 images are in color
        if cmap:
            plt.imshow(images[i], cmap=cmap)
        else:
            plt.imshow(images[i])
        plt.xlabel(labels[i])
    plt.show()


# Mappatura delle etichette CIFAR-10 in nomi di classe per una migliore leggibilità
cifar_class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                     'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Converti etichette CIFAR da numeri a nomi di classi
cifar_labels_readable = [cifar_class_names[label[0]] for label in cifar_train_labels[:25]]

# Visualizza le prime 25 immagini di MNIST
display_images(mnist_train_images[:25], mnist_train_labels[:25], title="MNIST Sample Images", cmap='gray')

# Visualizza le prime 25 immagini di CIFAR-10
display_images(cifar_train_images[:25], cifar_labels_readable, title="CIFAR-10 Sample Images")
