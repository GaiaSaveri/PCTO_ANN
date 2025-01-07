import tensorflow as tf

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