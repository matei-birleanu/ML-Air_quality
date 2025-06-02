
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    criterion: str = 'gini',
    class_weight: Optional[Dict[Any, float]] = None,
    max_samples: Optional[float] = None,
    max_features: str = 'auto',
    random_state: int = 42
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }


def document_random_forest(
    model: RandomForestClassifier
) -> Dict[str, Any]:
    params = model.get_params()
    depths      = [est.get_depth() for est in model.estimators_]
    node_counts = [est.tree_.node_count for est in model.estimators_]
    leaf_counts = [(est.tree_.children_left == -1).sum() for est in model.estimators_]

    return {
        'param_n_estimators':    params['n_estimators'],
        'param_max_depth':       params['max_depth'],
        'param_min_samples_leaf':params['min_samples_leaf'],
        'param_criterion':       params['criterion'],
        'param_class_weight':    params['class_weight'],
        'param_max_samples':     params.get('max_samples'),
        'param_max_features':    params['max_features'],
        'learned_avg_depth':     float(np.mean(depths)),
        'learned_avg_n_nodes':   float(np.mean(node_counts)),
        'learned_avg_n_leaves':  float(np.mean(leaf_counts)),
    }
