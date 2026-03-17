import numpy as np


c1 = np.loadtxt("Iris files/class_1", delimiter=",")
c2 = np.loadtxt("Iris files/class_2", delimiter=",")
c3 = np.loadtxt("Iris files/class_3", delimiter=",")


def split_data1(c1, c2, c3):
    X_train = np.vstack((c1[:30], c2[:30], c3[:30]))
    X_test  = np.vstack((c1[30:], c2[30:], c3[30:]))

    y_train = np.array([0]*30 + [1]*30 + [2]*30)
    y_test  = np.array([0]*20 + [1]*20 + [2]*20)

    return X_train, X_test, y_train, y_test

def split_data2(c1, c2, c3):
    X_train = np.vstack((c1[20:], c2[20:], c3[20:]))
    X_test  = np.vstack((c1[:20], c2[:20], c3[:20]))

    y_train = np.array([0]*30 + [1]*30 + [2]*30)
    y_test  = np.array([0]*20 + [1]*20 + [2]*20)

    return X_train, X_test, y_train, y_test

def one_hot_encoder(y, num_classes=3):
    T = np.zeros((len(y), num_classes))
    T[np.arange(len(y)), y] = 1
    return T

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_classifier(X, T, alpha, epochs):
    ant_samp= X.shape[0]
    ant_feat = X.shape[1]
    ant_class = T.shape[1]


    X_bias = np.hstack((X, np.ones((ant_samp,1))))

    
    W = np.random.randn(ant_feat+1, ant_class) * 0.01

    for epoch in range(epochs):
        grad = np.zeros_like(W)
        loss = 0

        for i in range(ant_samp):
            x = X_bias[i].reshape(-1,1)   # (d+1,1)
            t = T[i].reshape(-1,1)    # (K,1)

            z = np.dot(W.T, x)              # (K,1)
            g = sigmoid(z)

            error = g - t
            loss += np.sum(error**2)

            # gradient 
            grad += np.dot(x, (error * g * (1 - g)).T)

        grad =grad/ant_samp
        loss =loss/(2*ant_samp)

        W = W - alpha * grad

    return W

def predict(X, W):
    X_bias = np.hstack((X, np.ones((X.shape[0],1))))
    z = np.dot(X_bias, W)
    g = sigmoid(z)
    return np.argmax(g, axis=1)

def confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


X_train1, X_test1, y_train1, y_test1 = split_data1(c1, c2, c3)

T_train = one_hot_encoder(y_train1)

W = train_classifier(X_train1, T_train, alpha=0.9, epochs=1000)


y_train_pred1 = predict(X_train1, W)
y_test_pred1  = predict(X_test1, W)

print("Train error test 1:", error_rate(y_train1, y_train_pred1))
print("Test error test 1:", error_rate(y_test1, y_test_pred1))

print("Train confusion matrix test 1:\n", confusion_matrix(y_train1, y_train_pred1))
print("Test confusion matrix test 1:\n", confusion_matrix(y_test1, y_test_pred1))


X_train2, X_test2, y_train2, y_test2 = split_data2(c1, c2, c3)

T_train = one_hot_encoder(y_train2)

W = train_classifier(X_train2, T_train, alpha=0.9, epochs=1000)


y_train_pred2 = predict(X_train2, W)
y_test_pred2  = predict(X_test2, W)

print("Train error test 2:", error_rate(y_train2, y_train_pred2))
print("Test error test 2:", error_rate(y_test2, y_test_pred2))

print("Train confusion matrix test 2:\n", confusion_matrix(y_train2, y_train_pred2))
print("Test confusion matrix test 2:\n", confusion_matrix(y_test2, y_test_pred2))