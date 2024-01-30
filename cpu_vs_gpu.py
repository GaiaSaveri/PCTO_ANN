import tensorflow as tf
import time
from utils import tempo_esecuzione


n_gpus = len(tf.config.list_physical_devices('GPU'))
print("Num GPUs Disponibili: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(False)

# mettiamo le matrici nella CPU
with tf.device('/CPU:0'):
  cpu_a = tf.ones([15000, 15000])
  cpu_b = tf.zeros([15000, 15000])

# calcoliamo il tempo dell'esecuzione in CPU
print('device di a: ', cpu_a.device, '\ndevice di b: ', cpu_b.device)
cpu_inizio = time.time()
for _ in range(1):
  cpu_molt = tf.matmul(cpu_a, cpu_b)
cpu_fine = time.time()
tempo_esecuzione(cpu_inizio, cpu_fine, s=True)

if n_gpus > 0:
    # mettiamo le matrici nella GPU
    with tf.device('/GPU:0'):
      gpu_a = tf.ones([15000, 15000])
      gpu_b = tf.zeros([15000, 15000])

    # calcoliamo il tempo dell'esecuzione in CPU
    print('device di a: ', gpu_a.device, '\ndevice di b: ', gpu_b.device)
    gpu_inizio = time.time()
    for _ in range(1):
      gpu_molt = tf.matmul(gpu_a, gpu_b)
    gpu_fine = time.time()
    tempo_esecuzione(gpu_inizio, gpu_fine, s=True)
