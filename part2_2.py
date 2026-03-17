import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

""" def load_mnist(images_path, labels_path):
    images = np.fromfile(images_path, dtype=np.uint8)[16:].reshape(-1, 28*28)
    labels = np.fromfile(labels_path, dtype=np.uint8)[8:]
    return images, labels

testImages, testLabels = load_mnist("numbersdata/test_images.bin", "numbersdata/test_labels.bin")
trainImages, trainLabels = load_mnist("numbersdata/train_images.bin", "numbersdata/train_labels.bin")

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)

M = 64 """

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

    print(confusion_matrix(testLabels, predictions))
    return accuracy, confusion_matrix(testLabels, predictions)

