import tensorflow as tf
from tf.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Carica un'immagine ad alta risoluzione (sostituire con il percorso della tua immagine)
image_path = os.getcwd() + os.path.sep + 'image.png'
image = load_img(image_path)  # Carica l'immagine come PIL Image
image = img_to_array(image)  # Converte l'immagine in un array NumPy
image = np.expand_dims(image, axis=0)  # Aggiunge una dimensione per adattarsi all'API del generatore

# Definisce le trasformazioni per la data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,  # Gradi massimi di rotazione casuale dell'immagine
    width_shift_range=0.2,  # Frazione della larghezza totale per la traslazione orizzontale
    height_shift_range=0.2,  # Frazione dell'altezza totale per la traslazione verticale
    shear_range=0.2,  # Intensità di taglio (shear) per la trasformazione di taglio
    zoom_range=0.2,  # Range per il zoom casuale all'interno dell'immagine
    horizontal_flip=True,  # Capovolge casualmente le immagini orizzontalmente
    fill_mode='nearest'  # La strategia per riempire i nuovi pixel che possono apparire dopo una rotazione o un cambio di larghezza/altezza
)

# Funzione per visualizzare le immagini
def plot_images(images_arr, n_cols=4):
    n_rows = len(images_arr) // n_cols + (len(images_arr) % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i, ax in enumerate(axes.flat):
        if i < len(images_arr):
            ax.imshow(images_arr[i].astype('uint8'))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Genera immagini trasformate e le visualizza
augmented_images = []
for _ in range(8):
    augmented_img = datagen.random_transform(image[0])  # Applica una trasformazione casuale
    augmented_images.append(augmented_img)

plot_images(augmented_images)