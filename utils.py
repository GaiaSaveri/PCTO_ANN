import numpy as np
import matplotlib.pyplot as plt


def tempo_esecuzione(inizio, fine, s=False):
    ore, resto = divmod(fine - inizio, 3600)
    minuti, secondi = divmod(resto, 60)
    if s:
        print("Tempo di esecuzione = {:.4f}:{:.4f}:{:.4f}".format(int(ore), int(minuti), int(secondi)))
    return int(ore), int(minuti), int(secondi)


# funzione per visualizzare n immagini random dal dataset di input
def visualizza_immagini(dataset, y=None, n=10, nome_classi=None, colors='viridis'):
    viz_idx = np.random.choice(dataset.shape[0], n)
    fig, axs = plt.subplots(1, n, figsize=(15,3))
    for i in range(n):
        axs[i].imshow(dataset[viz_idx[i]], cmap=colors)
        axs[i].axis('off')
        if nome_classi is not None and y is not None:
            axs[i].set_title(nome_classi[y[viz_idx[i]]])
    plt.tight_layout()
    plt.show()
