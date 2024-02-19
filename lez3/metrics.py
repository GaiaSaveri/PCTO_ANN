import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# Funzione per caricare i dati salvati
def load_data(filename='cifar_results.pkl'):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


# Funzione per visualizzare i boxplot
def plot_boxplots(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=data[['loss', 'val_loss']], ax=axes[0])
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Dataset Type')

    sns.boxplot(data=data[['accuracy', 'val_accuracy']], ax=axes[1])
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Dataset Type')

    plt.tight_layout()
    plt.show()


# Funzione per la matrice di confusione
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# Funzione per calcolare e stampare la precisione per classe
def print_class_accuracy(true_labels, predicted_labels, classes):
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall accuracy: {accuracy * 100:.2f}%\n")
    print("Accuracy per class:")
    for i, class_name in enumerate(classes):
        class_accuracy = np.mean([true_labels == predicted_labels][0][true_labels == i])
        print(f"{class_name}: {class_accuracy * 100:.2f}%")


# Caricare i dati
data = load_data()

# Plot dei boxplot per loss e accuracy
plot_boxplots(pd.DataFrame(data['history']))

# Matrice di confusione
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes
plot_confusion_matrix(data['true_labels'], data['predicted_labels'], classes)

# Precisione per classe
print_class_accuracy(data['true_labels'], data['predicted_labels'], classes)
