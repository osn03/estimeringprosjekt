import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def train_kmeans(train_images, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_images)
    Ci = kmeans.cluster_centers_
    return Ci

def nearest_neighbor(test_images,test_labels, all_centers, center_labels):
    batch_size = 500
    predictions = np.empty(len(test_images), dtype=np.uint8)

    for start in range(0, len(test_images), batch_size):
        end = min(start + batch_size, len(test_images))
        batch = test_images[start:end]
        distances = cdist(batch, all_centers, metric='sqeuclidean')
        nearest_indices = np.argmin(distances, axis=1)
        predictions[start:end] = center_labels[nearest_indices]
        print(f"Processed {end} / {len(test_images)} images")
    accuracy = np.mean(predictions == test_labels)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Error rate: {(1 - accuracy):.2%}")
    return predictions, accuracy

#plot function with by chatGPT
def plot_confusion_matrix(cm, save_path="confusion_matrix_kmeans.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm)
    plt.colorbar(im)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix NN with clusters")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_task_2_2(testImages, testLabels, trainImages, trainLabels, M):
    centers = []
    center_labels = []

    for i in range(10):
        print("Training KMeans for digit", i)
        digit_images = trainImages[trainLabels == i]
        centers.append(train_kmeans(digit_images, M))
        center_labels.append(np.full(M, i))

    centers = np.vstack(centers)
    center_labels = np.hstack(center_labels)

    predictions, accuracy = nearest_neighbor(testImages, testLabels, centers, center_labels)   

    cm = confusion_matrix(testLabels, predictions)
    plot_confusion_matrix(cm)
    return accuracy, confusion_matrix(testLabels, predictions)

