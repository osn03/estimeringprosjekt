import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.spatial.distance import cdist


def load_mnist(images_path, labels_path):
    images = np.fromfile(images_path, dtype=np.uint8)[16:].reshape(-1, 28*28)
    labels = np.fromfile(labels_path, dtype=np.uint8)[8:]
    return images, labels

testImages, testLabels = load_mnist("numbersdata/test_images.bin", "numbersdata/test_labels.bin")
trainImages, trainLabels = load_mnist("numbersdata/train_images.bin", "numbersdata/train_labels.bin")

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)

#print("Test Images Shape:", testImages.shape)

#Euclidean distance function

def make_image(image_vector):
    return image_vector.reshape(28, 28)

def euclidean_distance(image_vector1, image_vector2):
    return np.sqrt(np.sum((image_vector1-image_vector2) ** 2))

def scipy_euclidean_distance(image_vector1, image_vector2):
    return sp.spatial.distance.euclidean(image_vector1, image_vector2)
# Test

def test_nearest_neighbor(test_images, test_labels, train_images, train_labels):
    correct_predictions = 0
    tests = test_images[:100]
    distances = cdist(tests, train_images, metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    predictions = train_labels[nearest_indices]
    correct_predictions = np.sum(predictions == test_labels[:100]) 
    accuracy = correct_predictions / len(tests) 
    print(f"Accuracy: {accuracy:.2%}")

def nearest_neighbor(test_images,test_labels, train_images, train_labels):
    batch_size = 500
    predictions = np.empty(len(test_images), dtype=np.uint8)

    for start in range(0, len(test_images), batch_size):
        end = min(start + batch_size, len(test_images))
        batch = test_images[start:end]
        distances = cdist(batch, train_images, metric='sqeuclidean')
        nearest_indices = np.argmin(distances, axis=1)
        predictions[start:end] = train_labels[nearest_indices]
        print(f"Processed {end} / {len(test_images)} images")
    accuracy = np.mean(predictions == test_labels)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Error rate: {100 - accuracy * 100:.2f}%")
    return predictions

def print_confusion_matrix(predictions, true_labels):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true][pred] += 1
    print(confusion_matrix)

#test_nearest_neighbor(testImages, testLabels, trainImages, trainLabels)
predictions = nearest_neighbor(testImages, testLabels, trainImages, trainLabels)
print_confusion_matrix(predictions, testLabels)

#printing some correct and incorrect predictions
wrong_indices = np.where(predictions != testLabels)[0]
right_indices = np.where(predictions == testLabels)[0]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(5):
    idx = wrong_indices[i]
    axes[0, i].imshow(make_image(testImages[idx]), cmap="gray")
    axes[0, i].set_title(f"T={testLabels[idx]}, P={predictions[idx]}")
    axes[0, i].axis("off")

for i in range(5):
    idx = right_indices[i]
    axes[1, i].imshow(make_image(testImages[idx]), cmap="gray")
    axes[1, i].set_title(f"T={testLabels[idx]}, P={predictions[idx]}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
