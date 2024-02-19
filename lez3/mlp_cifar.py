# Importazione delle librerie necessarie
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Caricamento del dataset CIFAR-10
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalizzazione delle immagini per avere valori tra 0 e 1
cifar_train_images = cifar_train_images / 255.0
cifar_test_images = cifar_test_images / 255.0

# Appiattimento delle immagini da 32x32x3 a 3072 per l'input del MLP
train_images_flattened = cifar_train_images.reshape((cifar_train_images.shape[0], 32 * 32 * 3))
test_images_flattened = cifar_test_images.reshape((cifar_test_images.shape[0], 32 * 32 * 3))

# Definizione del modello MLP
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)),  # Strato nascosto più grande a causa della complessità maggiore
    layers.Dropout(0.5),  # Aumento del dropout per combattere l'overfitting
    layers.Dense(10, activation='softmax')  # Uso di softmax per la classificazione multiclasse
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
history = model.fit(train_images_flattened, cifar_train_labels, epochs=50, validation_data=(test_images_flattened, cifar_test_labels))

# Valutazione del modello sul set di test
test_loss, test_accuracy = model.evaluate(test_images_flattened, cifar_test_labels, verbose=2)
print(f'\nTest Accuracy: {test_accuracy}')

# Visualizzazione dell'accuratezza durante l'addestramento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
