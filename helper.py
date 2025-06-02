from __future__ import annotations  # Necesar pentru a folosi tipul clasei în definiția ei
DATASET_ROOT = 'D:\IA\TEMA2CA'  # directorul în care se află seturile de date
DATASET_NAME = 'air_pollution__train.csv' 

from typing import Optional, Dict,List
from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


def save_table_to_text(
    summary_df: pd.DataFrame,
    filepath: str
):
    text = summary_df.to_string()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def load_dataset(dataset_filename: str) -> pd.DataFrame:
    """
    Incarca in memorie un set de date
    
    Args:
        dataset_filename (str): 
            Numele fisierului ce contine setul de date (cu tot cu extensie)
     
    Returns:
        pd.DataFrame: 
            Un DataFrame pandas ce contine setul de date
    """
    print (f"Dataset: {dataset_filename}")

    dataset_path = Path(DATASET_ROOT) /dataset_filename
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_filename} not found at {dataset_path}")
    return pd.read_csv(dataset_path)

def display_dataset(dataset: pd.DataFrame):
    """
    Afiseaza primele 5 intrări din setul de date
    
    Args:
        dataset (pd.DataFrame): 
            Setul de date in format pandas DataFrame.
    """
    print(dataset.head(n=5))
    
def display_dataset_feature_values(dataset: pd.DataFrame, feature: str):
    """
    Afiseaza valorile distincte ale unui atribut din setul de date
    
    Args:
        dataset (pd.DataFrame):
            Setul de date in format pandas DataFrame.
        feature (str):
            Numele atributului pentru care se vor afisa valorile distincte.
    """
    print(dataset[feature].unique())
    
def split_dataset(dataset: pd.DataFrame, target_feature: str) -> (pd.DataFrame, pd.Series):
    """
    Imparte setul de date in atribute si clase (etichete). In cazul seturilor noastre de date, ultima coloana reprezinta intotdeauna clasa.
    
    Args:
        dataset (pd.DataFrame): 
            Setul de date in format pandas DataFrame.
        target_feature (str): 
            Numele atributului care reprezinta clasa.
        
    Returns:
        tuple(pd.DataFrame, pd.Series): 
            Un tuplu ce contine atributele si clasele setului de date in formatul (X, y)
    """    
    return dataset.drop(columns=[target_feature]), dataset[target_feature]

def split_train_test(dataset: pd.DataFrame, 
                     target_feature: str, 
                     test_size: float = 0.2) -> (pd.DataFrame, pd.Series, 
                                                 pd.DataFrame, pd.Series):
    """
    Splits the dataset into a training set and a testing set.

    Args:
        dataset (pd.DataFrame): 
            Setul de date in format pandas DataFrame.
        target_feature (str): 
            Numle atributului care reprezinta clasa.
        test_size (float, optional): 
            Proportia setului de date care va fi folosita pentru testare. Defaults to 0.2.

    Returns:
        tuple(pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): 
            Un tuplu ce contine atributele si clasele setului de date de antrenare si de testare in formatul 
            (X_train, y_train, X_test, y_test)
    """
    #!pip install scikit-learn
    from sklearn.model_selection import train_test_split
    
    # Imparte setul de date in set de antrenare si set de testare
    train_set, test_set = train_test_split(dataset, test_size=test_size, shuffle=True)
    
    # Imparte setul de date in atribute si clase
    X_train_, y_train_ = split_dataset(train_set, target_feature)
    X_test_, y_test_ = split_dataset(test_set, target_feature)

    return X_train_, y_train_, X_test_, y_test_


def identify_attribute_types(
    df: pd.DataFrame,
    discrete_threshold: int = 10,
    ordinal_features: Optional[List[str]] = None
) -> Dict[str, List[str]]:

    if ordinal_features is None:
        ordinal_features = []
    
    continuous_cols = []
    discrete_cols   = []
    ordinal_cols    = []
    
    for col in df.columns:
        ser = df[col]
        if col in ordinal_features:
            ordinal_cols.append(col)
            continue
        
        if pd.api.types.is_categorical_dtype(ser) and ser.cat.ordered:
            ordinal_cols.append(col)
            continue
        
        if pd.api.types.is_float_dtype(ser):
            continuous_cols.append(col)
        
        elif pd.api.types.is_integer_dtype(ser):
            if ser.nunique() > discrete_threshold:
                continuous_cols.append(col)
            else:
                discrete_cols.append(col)
        else:
            discrete_cols.append(col)
    
    return {
        'continuous': continuous_cols,
        'discrete': discrete_cols,
        'ordinal': ordinal_cols
    }



def summarize_continuous_attributes(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None
) -> pd.DataFrame:

    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    summary = pd.DataFrame(index=index, columns=continuous_cols, dtype=float)

    # pentru fiecare coloana calculez statisticile
    for col in continuous_cols:
        series = df[col]
        cnt = series.count()  
        avg = series.mean()
        std = series.std()
        min_val = series.min()
        q25 = series.quantile(0.25)
        q50 = series.quantile(0.50)
        q75 = series.quantile(0.75)
        max_val = series.max()

        summary.at['count', col] = cnt
        summary.at['mean',  col] = avg
        summary.at['std',   col] = std
        summary.at['min',   col] = min_val
        summary.at['25%',   col] = q25
        summary.at['50%',   col] = q50
        summary.at['75%',   col] = q75
        summary.at['max',   col] = max_val

    return summary


def plot_continuous_boxplots(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    figsize: tuple[int,int] = (10, 6)
):
    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ax = df[continuous_cols].boxplot(figsize=figsize)
    plt.xticks(rotation=45, ha='right')
    plt.title("Boxplot pentru variabile numerice continue")
    plt.tight_layout()
    plt.show()


def summarize_discrete_attributes(
    df: pd.DataFrame,
    discrete_cols: list[str] | None = None
) -> pd.DataFrame:
    
    if discrete_cols is None:
        discrete_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    index = ['count', 'unique']
    summary = pd.DataFrame(index=index, columns=discrete_cols, dtype=int)

    for col in discrete_cols:
        series = df[col]
        cnt = series.count()

        uniq = series.nunique(dropna=True)

        summary.at['count',  col] = cnt
        summary.at['unique', col] = uniq

    return summary

def plot_categorical_histograms(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    top_n: int = 10,
    figsize: tuple[int,int] = (8, 6)
):

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        
        top = counts.iloc[:top_n].copy()
        others = counts.iloc[top_n:].sum()
        if others > 0:
            top['Other'] = others
        
        ax = top.sort_values().plot.barh(figsize=figsize)
        ax.set_title(f"Top {top_n} + Other pentru '{col}'")
        ax.set_xlabel("Frecvența")
        ax.set_ylabel(col)
        for spine in ['top','right','left']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.show()


def plot_label_distribution_simple(
    y: pd.Series,
    use_seaborn: bool = False,
    width: float = 0.8,
    figsize: tuple[int,int] = (8,4)
):

    counts = y.value_counts(dropna=False).sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    if use_seaborn:
        sns.countplot(x=y, order=counts.index, ax=ax)
    else:
        counts.plot.bar(ax=ax, width=width)

    for patch in ax.patches:
        height = patch.get_height()
        pct = height / total * 100
        ax.annotate(
            f'{pct:.1f}%',
            xy=(patch.get_x() + patch.get_width()/2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom'
        )

    ax.set_title("Distribuția claselor")
    ax.set_xlabel("Clasă")
    ax.set_ylabel("Număr exemple")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def compute_numeric_correlation(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)
    return corr

def plot_numeric_correlation(df: pd.DataFrame, method: str = 'pearson', figsize: tuple[int,int] = (8,8)):
    corr = compute_numeric_correlation(df, method=method)
    plt.figure(figsize=figsize)
    plt.matshow(corr, fignum=plt.gcf().number)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(f'Corelație numerică ({method})', pad=20)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def compute_chi2_pairs(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    dropna: bool = True
) -> pd.DataFrame:

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    records = []
    for col1, col2 in combinations(categorical_cols, 2):
        table = pd.crosstab(df[col1], df[col2], dropna=dropna)
        chi2, p, dof, _ = chi2_contingency(table)
        records.append({
            'var1':    col1,
            'var2':    col2,
            'chi2':    chi2,
            'p_value': p,
            'dof':     dof
        })

    return pd.DataFrame(records)



def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:

    total = df.shape[0]
    missing_count = df.isna().sum()
    missing_percent = (missing_count / total) * 100

    summary = pd.DataFrame({
        'total': total,
        'missing_count': missing_count,
        'missing_percent': missing_percent.round(2)
    })
    return summary

def impute_univariate(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: list[str] | None = None
) -> pd.DataFrame:

    X = df.copy()
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    imputer = SimpleImputer(strategy=strategy)
    X[columns] = imputer.fit_transform(X[columns])
    return X

def impute_multivariate(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    random_state: int = 0,
    max_iter: int = 10
) -> pd.DataFrame:

    X = df.copy()
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
    X[columns] = imputer.fit_transform(X[columns])
    return X



def mark_outliers_as_missing(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    lower_q: float = 0.25,
    upper_q: float = 0.75,
    factor: float = 1.5
) -> pd.DataFrame:
    X = df.copy()
    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for col in continuous_cols:
        s = X[col]
        q1 = s.quantile(lower_q)
        q3 = s.quantile(upper_q)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask = (s < lower) | (s > upper)
        X.loc[mask, col] = np.nan

    return X


def impute_outliers_univariate(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    lower_q: float = 0.25,
    upper_q: float = 0.75,
    factor: float = 1.5,
    strategy: str = 'median'
) -> pd.DataFrame:

    X = mark_outliers_as_missing(df, continuous_cols, lower_q, upper_q, factor)

    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    imputer = SimpleImputer(strategy=strategy)
    X[continuous_cols] = imputer.fit_transform(X[continuous_cols])

    return X


def impute_outliers_multivariate(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    lower_q: float = 0.25,
    upper_q: float = 0.75,
    factor: float = 1.5,
    random_state: int = 0,
    max_iter: int = 10
) -> pd.DataFrame:

    X = mark_outliers_as_missing(df, continuous_cols, lower_q, upper_q, factor)

    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
    X[continuous_cols] = imputer.fit_transform(X[continuous_cols])

    return X

def compute_iqr_bounds(
    s: pd.Series,
    q_lower: float = 0.25,
    q_upper: float = 0.75,
    factor: float = 1.5
) -> tuple[float, float]:
    q1, q3 = s.quantile(q_lower), s.quantile(q_upper)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr

def detect_extreme_values(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    q_lower: float = 0.25,
    q_upper: float = 0.75,
    factor: float = 1.5
) -> pd.DataFrame:
    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = pd.DataFrame(False, index=df.index, columns=continuous_cols)
    for col in continuous_cols:
        lower, upper = compute_iqr_bounds(df[col], q_lower, q_upper, factor)
        mask[col] = df[col].lt(lower) | df[col].gt(upper)
    return mask

def mark_extremes_as_missing(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    q_lower: float = 0.25,
    q_upper: float = 0.75,
    factor: float = 1.5
) -> pd.DataFrame:
    X = df.copy()
    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    mask = detect_extreme_values(X, continuous_cols, q_lower, q_upper, factor)
    X[continuous_cols] = X[continuous_cols].mask(mask, np.nan)
    return X

def impute_extremes_univariate(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    q_lower: float = 0.25,
    q_upper: float = 0.75,
    factor: float = 1.5,
    strategy: str = 'median'
) -> pd.DataFrame:
    X = mark_extremes_as_missing(df, continuous_cols, q_lower, q_upper, factor)
    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    X[continuous_cols] = imputer.fit_transform(X[continuous_cols])
    return X

def impute_extremes_multivariate(
    df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    q_lower: float = 0.25,
    q_upper: float = 0.75,
    factor: float = 1.5,
    random_state: int = 0,
    max_iter: int = 10
) -> pd.DataFrame:
    X = mark_extremes_as_missing(df, continuous_cols, q_lower, q_upper, factor)
    if continuous_cols is None:
        continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
    X[continuous_cols] = imputer.fit_transform(X[continuous_cols])
    return X

def standardize_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    X = df.copy()
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X[columns] = scaler.fit_transform(X[columns])
    return X



def load_and_split(
    dataset_filename: str,
    target_feature: str,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    stratify: bool = True
) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_filename)
    stratify_col = df[target_feature] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_col
    )
    X_train = train_df.drop(columns=[target_feature])
    y_train = train_df[target_feature]
    X_test  = test_df.drop(columns=[target_feature])
    y_test  = test_df[target_feature]
    return X_train, y_train, X_test, y_test