import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def print_confusion_matrix(predictions, true_labels):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true][pred] += 1
    print(confusion_matrix)

def confusion_matrix(predictions, true_labels):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true][pred] += 1
    return confusion_matrix

#plot function with by chatGPT
def plot_confusion_matrix(true_labels, predictions, save_path="confusion_matrixKNN.png"):
    cm = np.zeros((10, 10), dtype=int)
    
    for t, p in zip(true_labels, predictions):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    plt.colorbar(im)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix KNN")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def k_nearest_neighbors(test_images, test_labels, train_images, train_labels, k):
    batch_size = 500
    predictions = np.empty(len(test_images), dtype=np.uint8)

    for start in range(0, len(test_images), batch_size):
        end = min(start + batch_size, len(test_images))
        batch = test_images[start:end]
        distances = cdist(batch, train_images, metric='sqeuclidean')

        nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
        nearest_labels = train_labels[nearest_indices]
        predictions[start:end] = np.array([np.bincount(labels).argmax() for labels in nearest_labels])
        print(f"Processed {end} / {len(test_images)} images")
    accuracy = np.mean(predictions == test_labels)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Error rate: {(1 - accuracy):.2%}")
    return predictions, accuracy

def run_KNN(testImages, testLabels, trainImages, trainLabels, K):
    predictions, accuracy = k_nearest_neighbors(testImages, testLabels, trainImages, trainLabels, K)
    plot_confusion_matrix(testLabels, predictions)
    return accuracy, confusion_matrix(predictions, testLabels)  