import KNN
import part2_1
import part2_2
import numpy as np
import time

def load_mnist(images_path, labels_path):
    images = np.fromfile(images_path, dtype=np.uint8)[16:].reshape(-1, 28*28)
    labels = np.fromfile(labels_path, dtype=np.uint8)[8:]
    return images, labels

testImages, testLabels = load_mnist("numbersdata/test_images.bin", "numbersdata/test_labels.bin")
trainImages, trainLabels = load_mnist("numbersdata/train_images.bin", "numbersdata/train_labels.bin")

trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
M = 64
K = 7

start = time.perf_counter()
accuracy_2_1, confusion_matrix_2_1 = part2_1.run_task_2_1(testImages, testLabels, trainImages, trainLabels)
time_2_1 = time.perf_counter() - start

start = time.perf_counter()
accuracy_2_2, confusion_matrix_2_2 = part2_2.run_task_2_2(testImages, testLabels, trainImages, trainLabels, M)
time_2_2 = time.perf_counter() - start

start = time.perf_counter()
accuracy_KNN, confusion_matrix_KNN = KNN.run_KNN(testImages, testLabels, trainImages, trainLabels, K)
time_KNN = time.perf_counter() - start

print(f"Task 2.1 (Nearest Neighbor) Accuracy: {accuracy_2_1:.2%}\t Time: {time_2_1:.2f} seconds")
print(f"Task 2.2 (KMeans + Nearest Neighbor) Accuracy: {accuracy_2_2:.2%}\t Time: {time_2_2:.2f} seconds")
print(f"KNN (K={K}) Accuracy: {accuracy_KNN:.2%}\t Time: {time_KNN:.2f} seconds")