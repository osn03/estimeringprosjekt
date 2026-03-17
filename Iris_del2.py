import matplotlib.pyplot as plt
import numpy as np
from Iris_del1 import train_classifier
from Iris_del1 import one_hot_encoder
from Iris_del1 import predict
from Iris_del1 import error_rate
from Iris_del1 import confusion_matrix



c1 = np.loadtxt("Iris files/class_1", delimiter=",")
c2 = np.loadtxt("Iris files/class_2", delimiter=",")
c3 = np.loadtxt("Iris files/class_3", delimiter=",")


def split_data1(c1, c2, c3):
    X_train = np.vstack((c1[:30], c2[:30], c3[:30]))
    X_test  = np.vstack((c1[30:], c2[30:], c3[30:]))

    y_train = np.array([0]*30 + [1]*30 + [2]*30)
    y_test  = np.array([0]*20 + [1]*20 + [2]*20)

    return X_train, X_test, y_train, y_test


feature_names = ["sepal length", "sepal width", "petal length", "petal width"]


"""
for i in range(4):
    plt.figure()

    plt.hist(c1[:, i], bins=10, alpha=0.5, label="Class 1")
    plt.hist(c2[:, i], bins=10, alpha=0.5, label="Class 2")
    plt.hist(c3[:, i], bins=10, alpha=0.5, label="Class 3")

    plt.title(f"Feature {i}: {feature_names[i]}")
    plt.legend()
    plt.show()
"""
X_train1, X_test1, y_train1, y_test1 = split_data1(c1, c2, c3)

X_train_3feat = np.delete(X_train1, 1, axis=1)
X_test_3feat= np.delete(X_test1, 1, axis=1)
T_train = one_hot_encoder(y_train1)

W_3feat = train_classifier(X_train_3feat, T_train, alpha=0.9, epochs=1000)

y_train_pred_3feat = predict(X_train_3feat, W_3feat)
y_test_pred1_3feat  = predict(X_test_3feat, W_3feat)

print("Train error with 3 features:", error_rate(y_train1, y_train_pred_3feat))
print("Test error with 3 features:", error_rate(y_test1, y_test_pred1_3feat))

print("Train confusion matrix with 3 features:\n", confusion_matrix(y_train1, y_train_pred_3feat))
print("Test confusion matrix with 3 features:\n", confusion_matrix(y_test1, y_test_pred1_3feat))

X_train_2feat = X_train1[:,[2,3]]
X_test_2feat= X_test1[:,[2,3]]


W_2feat = train_classifier(X_train_2feat, T_train, alpha=0.9, epochs=1000)

y_train_pred_2feat = predict(X_train_2feat, W_2feat)
y_test_pred1_2feat  = predict(X_test_2feat, W_2feat)

print("Train error with 2 features:", error_rate(y_train1, y_train_pred_2feat))
print("Test error with 2 features:", error_rate(y_test1, y_test_pred1_2feat))

print("Train confusion matrix with 2 features:\n", confusion_matrix(y_train1, y_train_pred_2feat))
print("Test confusion matrix with 2 features:\n", confusion_matrix(y_test1, y_test_pred1_2feat))

X_train_1feat = X_train1[:, [3]]  
X_test_1feat  = X_test1[:, [3]]


W_1feat = train_classifier(X_train_1feat, T_train, alpha=0.9, epochs=1000)

y_train_pred_1feat = predict(X_train_1feat, W_1feat)
y_test_pred1_1feat  = predict(X_test_1feat, W_1feat)

print("Train error with 1 features:", error_rate(y_train1, y_train_pred_1feat))
print("Test error with 1 features:", error_rate(y_test1, y_test_pred1_1feat))

print("Train confusion matrix with 1 features:\n", confusion_matrix(y_train1, y_train_pred_1feat))
print("Test confusion matrix with 1 features:\n", confusion_matrix(y_test1, y_test_pred1_1feat))