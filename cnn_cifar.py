# Importare le librerie necessarie
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle
from metrics import * 

# Verifica della disponibilità della GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurazione di TensorFlow per utilizzare la GPU se disponibile
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        # Gestione degli errori in caso di problemi di configurazione
        print(e)
else:
    print("GPU non disponibile, verrà utilizzata la CPU.")

# Caricare e preparare il dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalizzazione e reshaping
x_train_norm = # TODO: le immagini sono a colori, e hanno dimensione 32 x 32 x 3
x_test_norm = # TODO: come sopra
# One-hot-encoding delle etichette
y_train_enc = # TODO
y_test_enc = # TODO

lr = 0.001
# Definizione della funzione per creare il modello
def create_optimized_model():
    model = models.Sequential([
        # per tutte le convoluzioni, useremo padding='same', activation='relu'
        # Primo blocco convoluzionale
        # conv: 32 filtri, kernel_size=(3, 3), input_shape=(32, 32, 3)

        # batch normalization

        # conv: 32 filtri, kernel_size=(3, 3)

        # batch normalization

        # max pooling (senza argomenti)

        # dropout: 0.2


        # Secondo blocco convoluzionale
        # conv: 64 filtri, kernel_size=(3, 3)

        # batch normalization

        # conv: 64 filtri, kernel_size=(3, 3)

        # batch normalization

        # max pooling (senza argomenti)

        # dropout: 0.3


        # Terzo blocco convoluzionale
        # conv: 128 filtri, kernel_size=(3, 3)

        # batch normalization

        # conv: 128 filtri, kernel_size=(3, 3)

        # batch normalization

        # max pooling (senza argomenti)

        # dropout: 0.4


        # Strato di appiattimento e strato denso finale
        # flatten

        # dense: 128 neuroni, activation='relu'

        # batch normalization

        # dropout: 0.5

        # dense: 10 neuroni, activation='softmax'

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


# Creare e addestrare il modello ottimizzato
model = create_optimized_model()
n_epochs = 10
history = model.fit(x_train_norm, y_train_enc, epochs=10, validation_split=0.2)

# Valutazione del modello
test_loss, test_acc = # TODO: evaluate
print('\nTest accuracy with subset:', test_acc)

# Utilizzare il modello per fare previsioni sul subset di test
y_pred_prob = # TODO: predict
y_pred = np.argmax(y_pred_prob, axis=1)

# Salvare le metriche, le etichette vere e predette
results = {
    'history': history.history,
    'test_loss': test_loss,
    'test_accuracy': test_acc,
    'true_labels': y_test.flatten(),
    'predicted_labels': y_pred,
    'predicted_probabilities': y_pred_prob,
    'true_features': x_test_norm
}

with open(f'cnn_cifar_{n_epochs}_epochs_{lr}_lr.pkl', 'wb') as file:
    pickle.dump(results, file)

# TODO: plot con le funzioni definite in metrics.py