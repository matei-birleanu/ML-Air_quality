from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def train_mlp_classifier(
    X_train, y_train,
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    alpha=0.0001,
    max_iter=200,
    batch_size='auto',
    early_stopping=True,
    random_state=42
):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        batch_size=batch_size,
        early_stopping=early_stopping,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_mlp_classifier(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return {
        'accuracy': acc,
        'classification_report': report,
        'preds': preds
    }


def plot_mlp_confusion_matrix(model, X_test, y_test, labels):
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=labels,
        cmap="Blues",
        normalize='true'
    )
    disp.figure_.suptitle("MLP - Confusion Matrix (normalized)")
    plt.tight_layout()
    plt.show()
