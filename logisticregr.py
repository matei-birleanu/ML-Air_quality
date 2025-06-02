import numpy as np
import math

def logistic(z):

    if isinstance(z, (float, int)):
        return 1.0 / (1.0 + math.exp(-z))
    else:
        arr = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + np.exp(-arr))


def nll(Y: np.ndarray, T: np.ndarray) -> float:
    N = T.shape[0]
    eps = 1e-12
    Yc = np.clip(Y, eps, 1 - eps)
    return - (T * np.log(Yc) + (1 - T) * np.log(1 - Yc)).mean()


def accuracy(Y: np.ndarray, T: np.ndarray) -> float:
    preds = (Y >= 0.5).astype(int)
    return (preds == T).mean()


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return logistic(X @ w)


def train_and_eval_logistic(
    X_train: np.ndarray,
    T_train: np.ndarray,
    X_test:  np.ndarray,
    T_test:  np.ndarray,
    lr: float = 0.01,
    epochs: int = 100,
    reg_strength: float = 0.0
):
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    T_train = np.asarray(T_train, dtype=int)
    T_test  = np.asarray(T_test,  dtype=int)

    N, D = X_train.shape
    w = np.zeros(D, dtype=float)

    train_nll, test_nll = [], []
    train_acc, test_acc = [], []

    for _ in range(epochs):
        Y_train = predict_logistic(X_train, w)
        Y_test  = predict_logistic(X_test,  w)

        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test,  T_test))
        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test,  T_test))

        grad = (X_train.T @ (Y_train - T_train)) / N
        if reg_strength > 0:
            grad += reg_strength * w
        w -= lr * grad

    return w, train_nll, test_nll, train_acc, test_acc
