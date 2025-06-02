from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional, Dict, Any
from typing import Optional, List, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import ClassifierMixin

def train_decision_tree(
    X_train, 
    y_train, 
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    criterion: str = 'gini',
    class_weight: Optional[Dict[Any, float]] = None,
    random_state: int = 0
) -> DecisionTreeClassifier:

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_decision_tree(
    model: DecisionTreeClassifier,
    X_test, 
    y_test
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {
        'accuracy': acc,
        'classification_report': report
    }


def get_tree_info(model: DecisionTreeClassifier) -> Dict[str, Any]:

    info = {
        'param_max_depth':         model.max_depth,
        'param_min_samples_leaf':  model.min_samples_leaf,
        'param_criterion':         model.criterion,
        'param_class_weight':      model.class_weight,
        'learned_depth':           model.get_depth(),
        'learned_n_nodes':         model.tree_.node_count,
        'learned_n_leaves':        model.get_n_leaves()
    }
    return info


def plot_confusion_matrix(
    model: ClassifierMixin,
    X, 
    y, 
    labels: Optional[List[Any]] = None,
    normalize: Optional[str] = None,
    figsize: Tuple[int,int] = (6,6),
    cmap: str = 'Blues'
):
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X, y,
        display_labels=labels,
        normalize=normalize,
        cmap=cmap,
        xticks_rotation='horizontal',
        colorbar=True,
    )
    disp.figure_.set_size_inches(*figsize)
    plt.title("Matrice de confuzie" + (f" (norm: {normalize})" if normalize else ""))
    plt.tight_layout()
    plt.show()