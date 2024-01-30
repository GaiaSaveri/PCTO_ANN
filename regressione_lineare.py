import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


def modello_lineare(learning_rate):
    modello = tf.keras.models.Sequential()
    modello.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    modello.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="mean_squared_error",
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return modello


def allena_modello(modello, dati, etichette, epoche, batch_size):
    history = modello.fit(x=dati, y=etichette, batch_size=batch_size, epochs=epoche)
    pesi_finali = modello.get_weights()[0]
    correzioni_finali = modello.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]  # errore ad ogni epoca
    return pesi_finali, correzioni_finali, epochs, rmse


def mostra_modello(pesi, correzioni, dati, etichette):
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(dati, etichette)
    x0 = 0
    y0 = correzioni
    x1 = dati[-1]
    y1 = (correzioni + (pesi * x1))[0]
    plt.plot(np.array([x0, x1]), np.array([y0, y1]), c='r')
    plt.show()


def visualizza_costo(epoche, rmse):
    plt.figure()
    plt.xlabel("Epoca")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(epoche, rmse, label="Costo")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()


lista_x = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
lista_y = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

lr = 0.01
n_epoche = 250
n_batch = 12

reg_lin = modello_lineare(lr)
pesi_reg_lin, correzioni_reg_lin, epochs_rec, rmse_rec = allena_modello(reg_lin, lista_x, lista_y, n_epoche, n_batch)
mostra_modello(pesi_reg_lin, correzioni_reg_lin, lista_x, lista_y)
visualizza_costo(epochs_rec, rmse_rec)
