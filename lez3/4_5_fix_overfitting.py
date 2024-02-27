import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

# Download e preparazione del dataset
dataset = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000)


#TOCCA A VOI
