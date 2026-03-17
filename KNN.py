import numpy as np
from scipy.spatial.distance import cdist

def load_mnist(images_path, labels_path):
    images = np.fromfile(images_path, dtype=np.uint8)[16:].reshape(-1, 28*28)
    labels = np.fromfile(labels_path, dtype=np.uint8)[8:]
    return images, labels

testImages, testLabels = load_mnist("numbersdata/test_images.bin", "numbersdata/test_labels.bin")
trainImages, trainLabels = load_mnist("numbersdata/train_images.bin", "numbersdata/train_labels.bin")

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)

def print_confusion_matrix(predictions, true_labels):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true][pred] += 1
    print(confusion_matrix)

K = 7

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
    return predictions
predictions = k_nearest_neighbors(testImages, testLabels, trainImages, trainLabels, K)
print_confusion_matrix(predictions, testLabels)