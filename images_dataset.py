import tensorflow as tf
from utils import visualizza_immagini
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


nome_dataset = 'CIFAR'  # da scegliere tra mnsit e cifar
if nome_dataset == 'MNIST':
    # carichiamo il dataset MNIST (già presente in tensorflow)
    # il dataset è già diviso in training e test set
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

    # controlliamo le dimensioni del dataset
    print('Dimensioni dataset di train (MNIST): ', x_train_mnist.shape, y_train_mnist.shape)
    print('Dimensioni dataset di test (MNIST): ', x_test_mnist.shape, y_test_mnist.shape)
    mnist_y = [str(i) for i in range(10)]
    visualizza_immagini(x_train_mnist, y=y_train_mnist, nome_classi=mnist_y, colors='gray')
    print('Minimo valore intensita: ', x_train_mnist.min())
    print('Massimo valore intensita: ', x_train_mnist.max())
    # one-hot encoding delle etichette (così sono vettori con tanti elementi quante le possibili classi)
    y_train_onehot_mnist = tf.keras.utils.to_categorical(y_train_mnist)
    y_test_onehot_mnist = tf.keras.utils.to_categorical(y_test_mnist)
    print("Prime 3 etichette di training: ", y_train_mnist[:3])
    print("Etichette one-hot-encoded dei primi 3 punti di train:\n", y_train_onehot_mnist[:3])
    print("Nuova dimensione etichette dopo one-hot-encoding: ", y_train_onehot_mnist.shape, y_test_onehot_mnist.shape)

else:  # CIFAR
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()
    print('Dimensioni dataset di train (Cifar10): ', x_train_cifar.shape, y_train_cifar.shape)
    cifar_y = ['airplane', 'automobile', 'bird', 'cat', 'deer',  'dog', 'frog', 'horse', 'ship', 'truck']
    visualizza_immagini(x_train_cifar, y=tf.squeeze(y_train_cifar), nome_classi=cifar_y)
