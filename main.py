from __future__ import annotations  # Necesar pentru a folosi tipul clasei în definiția ei
DATASET_NAME = 'news_popularity_full.csv'

TARGET_FEATURE = 'popularity_category'
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from tqdm import tqdm
from helper import (
    load_and_split,
    load_dataset,
    display_dataset_feature_values,
    identify_attribute_types,
    split_dataset,
    split_train_test,

    summarize_continuous_attributes,
    plot_continuous_boxplots,

    summarize_discrete_attributes,
    plot_categorical_histograms,
    plot_label_distribution_simple,

    compute_numeric_correlation,
    plot_numeric_correlation,
    compute_chi2_pairs,

    missing_value_summary,
    impute_univariate,
    impute_multivariate,

    detect_extreme_values,
    mark_extremes_as_missing,
    impute_extremes_univariate,
    impute_extremes_multivariate,

    standardize_numeric,
    save_table_to_text
)
from decisiontrees import (
    train_decision_tree,
    evaluate_decision_tree,
    get_tree_info,
    plot_confusion_matrix
)
from logisticregr import (
    train_and_eval_logistic,
    predict_logistic
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from randomforest import (
    train_random_forest,
    evaluate_random_forest,
    document_random_forest
)
from mlp import(
    train_mlp_classifier, evaluate_mlp_classifier, plot_mlp_confusion_matrix
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    data = load_dataset(DATASET_NAME)

    types = identify_attribute_types(
        data,
        discrete_threshold=7,
        ordinal_features=['popularity_category']
    )
    print("Continuous:", types['continuous'])
    print("Discrete:  ", types['discrete'])
    print("Ordinal:   ", types['ordinal'])
    ##------------------------------------CERINTA 3.1.1 ---------------------------------------------##
    # ATRIBUTE CONTINUE
    summary = summarize_continuous_attributes(data,types['continuous'])
    print(summary)
    #salvez tabelul in format csv
    output_path = "full_step_news/tabel_continuous_attributes.csv"
    summary.to_csv(output_path, index=True)
    # salvez tabelul in format text
    save_table_to_text(summary,"tabel_text_continuous_attributes")
    
    plot_continuous_boxplots(data,types['continuous'])
    #ATRIBUTE DISCRETE SI ORDINALE 
    summary_discrete = summarize_discrete_attributes(data,types['discrete'] + types['ordinal'])
    output_path = "tabel_discrete_ord_attributes.csv"
    summary_discrete.to_csv(output_path)
    save_table_to_text(summary_discrete,"train_pollution/discreteord/text_discrete_ord_attributes")
    print(summary_discrete)
    plot_categorical_histograms(data,types['discrete'] + types['ordinal'])
    ##------------------------------------CERINTA 3.1.2 ---------------------------------------------##
    X_train, y_train = split_dataset(data, TARGET_FEATURE)
    plot_label_distribution_simple(y_train,use_seaborn=False)
    ##------------------------------------CERINTA 3.1.3 ---------------------------------------------##
    print("=== Matrice corelație numerică ===")
    numeric_corr = compute_numeric_correlation(data)
    save_table_to_text(numeric_corr,"full_step_news/matrice_numeric_corr")
    print(numeric_corr)

    plot_numeric_correlation(data)

    # corelatie categorica
    print("\n=== Teste Chi2 între categorice ===")
    chi2_df = compute_chi2_pairs(data)
    # afisez doar perechi semnificative
    significant = chi2_df[chi2_df['p_value'] < 0.05]
    print(significant)
    save_table_to_text(significant,"full_step_news/significant")


    ##------------------------------------CERINTA 3.2.1 ---------------------------------------------##
    miss = missing_value_summary(data)
    print("=== Missing values summary ===")
    save_table_to_text(miss,"full_step_news/missing_values_summ")
    print(miss)
    univariat_data = impute_univariate(data, strategy='median')
    multivar_data = impute_multivariate(data)
    univariat_data.to_csv("full_step_news/full_imputed_univariate.csv",index=False)
    multivar_data.to_csv("full_step_news/full_imputed_multivariate.csv",index=False)
    ##-----------------------------------CERINTA 3.2.2------------------------------------------------##
    df_for_extreme = load_dataset("full_step_news/full_imputed_univariate.csv")

    # verific cati outlieri detectez
    mask = detect_extreme_values(df_for_extreme, q_lower=0.25, q_upper=0.75, factor=1.5)
    print("Outliers detectați per coloană:")
    print(mask.sum())

    # marchez cu Nan
    df_marked = mark_extremes_as_missing(df_for_extreme, q_lower=0.25, q_upper=0.75, factor=1.5)
    df_marked.to_csv("full_step_news/full_marked_outliers.csv", index=False)

    # imputare univariata
    df_ext_uni = impute_extremes_univariate(
         df_marked,
         strategy='median',
         q_lower=0.25, q_upper=0.75, factor=1.5
     )
    df_ext_uni.to_csv("full_step_news/full_ext_uni_imputed.csv", index=False)

    # imputare multivariata
    df_ext_multi = impute_extremes_multivariate(
        df_marked,
        q_lower=0.25, q_upper=0.75, factor=1.5
    )
    df_ext_multi.to_csv("full_step_news/full_ext_multi_imputed.csv", index=False)

    # ##------------------------------------CERINTA 3.2.3 ---------------------------------------------##
    # coloane pe care le elimin din setul de date
    cols_to_drop = [
    'unique_non_stop_ratio',
    'non_stop_word_ratio',
    'keyword_worst_avg_shares',
    'keyword_avg_avg_shares',
    'ref_avg_shares',
    'content_density',
    'non_neutral_positive_rate',
    'max_positive_sentiment',
    'min_negative_sentiment',
    'max_negative_sentiment'
]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_ext_uni.columns]

    df_reduced = df_ext_uni.drop(columns=existing_cols_to_drop)
    ## modific intre df_ext_uni si df_ext_multi
    df_reduced = df_ext_uni.drop(columns=cols_to_drop)
    df_reduced.to_csv("full_step_news/full_selected_features.csv", index=False)
    ##------------------------------------CERINTA 3.2.4 ---------------------------------------------##
    final_dataset = load_dataset("full_step_news/full_selected_features.csv")
    x, y = split_dataset(final_dataset,TARGET_FEATURE)
    x_scaled = standardize_numeric(x)
    df_scaled = pd.concat([x_scaled, y], axis=1)
    df_scaled.to_csv("full_step_news/full_standardized_selected_features.csv", index=False)
    ##------------------------------------CERINTA 3.2.4 ---------------------------------------------##
    
    final_dataset = load_dataset("full_step_news/full_standardized_selected_features.csv")
   
    le = LabelEncoder()
    final_dataset[types['ordinal'][0]] = le.fit_transform(final_dataset[types['ordinal'][0]])
    final_dataset.drop(columns=['url'], inplace=True)
    types['discrete'].remove('url')
    # One-Hot encoding pe atributele discrete
    final_dataset = pd.get_dummies(
        final_dataset,
        columns=types['discrete'],
        drop_first=False
    )

    final_dataset.to_csv("encoded_news_full.csv", index=False)

    print("Saved encoded dataset to train_selected_features_encoded.csv")
    ##------------------------------------CERINTA 3.3.1 DECISION TREE---------------------------------------------##
    
    X_train, y_train, X_test, y_test = load_and_split(
        "encoded_news_full.csv",
        target_feature=TARGET_FEATURE,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=True
    )

    model = train_decision_tree(
        X_train,
        y_train,
        max_depth=5,
        min_samples_leaf=10,
        criterion='entropy',
        class_weight='balanced',
        random_state=42
    )
    ## scriu rezultatele in fisier
    with open("stats_news.txt", 'a', encoding='utf-8') as f:
        f.write("DECISION TREES\n")

        metrics = evaluate_decision_tree(model, X_test, y_test)

        line = f"Test accuracy : {metrics['accuracy']:.3f}"
        print(line)
        f.write(line + "\n\n")

        print("Classification report (test):")
        f.write("Classification report (test):\n")
        report = metrics['classification_report']
        print(report)
        f.write(report + "\n")

        info = get_tree_info(model)
        print("Decision Tree hyperparameters & learned stats:")
        f.write("Decision Tree hyperparameters & learned stats:\n")
        for name, val in info.items():
            line = f"  {name:<25}: {val}"
            print(line)
            f.write(line + "\n")
        f.write("\n\n\n")
    disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=sorted(y_test.unique()),
    cmap='Blues',
    normalize=None
)
    disp.figure_.set_size_inches(6, 6)
    plt.title("Confusion Matrix - Decision Tree (News Data)")
    plt.tight_layout()
    plt.show()
    #------------------------------------CERINTA 3.3.2 RANDOM FOREST---------------------------------------------##

    rf = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        criterion='entropy',
        class_weight='balanced',
        max_samples=0.8,
        max_features='sqrt',
        random_state=42
    )

    print("Random Forest")
    rf_info = document_random_forest(rf)
    print("\nRandom Forest hyperparameters & learned stats:")
    for name, val in rf_info.items():
        print(f"  {name:25}: {val}")

    metrics = evaluate_random_forest(rf, X_test, y_test)
    print(f"\nTest accuracy : {metrics['accuracy']:.3f}")
    print("Classification report (test):")
    print(metrics['classification_report'])

    # scriu in fisier sub accelasi format
    with open("stats.txt", "a", encoding="utf-8") as f:
        f.write("\nRandom Forest\n")
        f.write("Random Forest hyperparameters & learned stats:\n")
        for name, val in rf_info.items():
            f.write(f"  {name:25}: {val}\n")
        f.write(f"\nTest accuracy : {metrics['accuracy']:.3f}\n\n")
        f.write("Classification report (test):\n")
        f.write(metrics['classification_report'] + "\n")
    
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, rf.predict(X_test), labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    with open("stats.txt", "a", encoding="utf-8") as f:
        f.write("\nConfusion Matrix (count):\n")
        f.write(np.array2string(cm, separator=' ', max_line_width=1000) + "\n\n")
        f.write("Confusion Matrix (normalized by true):\n")
        f.write(np.array2string(cm_norm, precision=3, separator=' ', max_line_width=1000) + "\n")
    plot_confusion_matrix(
    rf,
    X_test,
    y_test,
    labels=labels,
    normalize=None,
    figsize=(6,6) )
    plot_confusion_matrix(
    rf,
    X_test,
    y_test,
    labels=labels,
    normalize='true',
    figsize=(6,6) )
     ##------------------------------------CERINTA 3.3.3 LOGISTIC REGRESSION ---------------------------------------------##
    X_train_mod = X_train.to_numpy(dtype=float)
    X_test_mod  = X_test.to_numpy(dtype=float)
    Y_train_mod = y_train.to_numpy(dtype=int)
    Y_test_mod  = y_test.to_numpy(dtype=int)
    w, train_nll, test_nll, train_acc, test_acc = train_and_eval_logistic(
        X_train_mod, Y_train_mod, X_test_mod, Y_test_mod,
        lr=0.01,
        epochs=200,
        reg_strength=0.001
    )
    ## valori de performanta 
    print(f"Final train NLL: {train_nll[-1]:.4f}")
    print(f"Final test  NLL: {test_nll[-1]:.4f}")
    print(f"Final train ACC: {train_acc[-1]:.4f}")
    print(f"Final test  ACC: {test_acc[-1]:.4f}")

    probs = predict_logistic(X_test, w)
    preds = (probs >= 0.5).astype(int)
    print("\nExemple de predicții (primele 10):")
    print("Probabilități:", np.round(probs[:10], 3))
    print("Predicții:   ", preds[:10])
    print("Adevărate:   ", Y_test_mod[:10])

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    y_true_named = label_encoder.inverse_transform(Y_test_mod)
    y_pred_named = label_encoder.inverse_transform(preds)
    acc = accuracy_score(y_true_named, y_pred_named)
    f1_macro = f1_score(y_true_named, y_pred_named, average='macro')
    report = classification_report(y_true_named, y_pred_named, digits=2)

    print(f"\nAccuracy : {acc:.3f}")
    print(f"F1_macro : {f1_macro:.2f}")
    print("\nRaport:\n")
    print(report)

    with open("stats_news.txt", "a", encoding="utf-8") as f:
        f.write("\nLogistic Regression — Etichete explicite\n")
        f.write(f"Accuracy : {acc:.3f}\n")
        f.write(f"F1_macro : {f1_macro:.2f}\n")
        f.write("\nRaport:\n")
        f.write(report)
        f.write("\n")
    labels = np.unique(Y_test_mod)

    disp_norm = ConfusionMatrixDisplay.from_predictions(
        Y_test_mod,
        preds,
        display_labels=labels,
        normalize='true',
        cmap='Blues'
    )
    disp_norm.figure_.suptitle("Logistic Regression — Confusion Matrix (normalized)")
    plt.show()
##------------------------------------CERINTA 3.3.4 MLP ---------------------------------------------##
    mlp = train_mlp_classifier(
    X_train, y_train,
    hidden_layer_sizes=(200,100, 50),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    alpha=0.0005,
    max_iter=300,
    batch_size=64,
    early_stopping=True,
    random_state=42
)   
    mlp_metrics = evaluate_mlp_classifier(mlp, X_test, y_test)
    print(f"\nMLP accuracy: {mlp_metrics['accuracy']:.3f}")
    print("Classification report (MLP):")
    print(mlp_metrics['classification_report'])
    with open("stats_news.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "="*40 + "\n")
        f.write("Multi-Layered Perceptron (MLP)\n")
        f.write("="*40 + "\n\n")
        f.write(f"MLP accuracy: {mlp_metrics['accuracy']:.3f}\n\n")
        f.write("Classification report (MLP):\n")
        f.write(mlp_metrics['classification_report'])
        f.write("\n")

    plot_mlp_confusion_matrix(mlp, X_test, y_test, labels=sorted(y_test.unique()))

if __name__ == "__main__":
    main()
