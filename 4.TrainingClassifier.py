# -*- coding: utf-8 -*-
##### Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.impute import KNNImputer
import xgboost as xgb
from bayes_xgb import *

__author__ = "Felix Dransfield"

# Helper functions

def KNNimputation(X, num_neighbours=10):
    imputer = KNNImputer(n_neighbors=num_neighbours)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imputed


def model_development(X, y, classifiers):
    scores_df = pd.DataFrame()
    for classifier in classifiers:
        print(f'Testing {classifier.__class__.__name__}')
        test_name = f'{classifier.__class__.__name__}'
        classifier.fit(X, y)
        scores = cross_validate(classifier, X, y, scoring="average_precision", cv=10, n_jobs=-1)
        scores_df[test_name] = scores['test_score']
    print("Testing completed")

    return scores_df

def main():
    """This script:
        1: Determines best classifier with 10-fold CV
        2: Uses Bayesian Optimization to optimize hyper-parameters
        3: Trains chosen algorithm with training data
        4: Evaluates on test data
    """
    ###### Creating Directory for outputs
    output_path = "4.CLASSIFIER OUTPUT"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##### Data import/Cleaning
    data = pd.read_csv("2.DATA CLEANING OUTPUT\cleaned_data.csv")
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    data.drop(['PatientID'], axis=1, inplace=True)
    data.columns = data.columns.str.replace("[", " ", regex=True)
    data.columns = data.columns.str.replace("]", " ", regex=True)

    # Train/Test Split (20% Test)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['diagnosis'], axis=1),
                                                        data['diagnosis'],
                                                        test_size=0.2,
                                                        random_state=7777
                                                        )

    # Version of training and testing data with missing values imputed
    X_train_imputed = KNNimputation(X_train)
    X_test_imputed = KNNimputation(X_test)

    classifier_list = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        SGDClassifier(),
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='aucpr'),
        GaussianNB(),
        MLPClassifier(alpha=1, max_iter=1000)
    ]

    scores_df = model_development(X_train_imputed, y_train, classifier_list)

    ##### Plotting results of model selection
    colours = ["#F0EFB0", "#F6A246", "#715E78", "#A5AAA3", "#812F33", "#0D7FA8", "#ABC472", "#288272"]
    sns.set_palette(sns.color_palette(colours))

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.boxplot(data=scores_df, palette=colours)
        plt.ylabel("Area under the Precision-Recall Curve", fontdict={'fontsize': 16})
        plt.xticks(rotation=45, fontsize=8)
        plt.show(block=False)
        plt.savefig(output_path + f"/Model selection", orientation='landscape')

    ###### Tuning hyper-parameters with Bayesian optimisation
    # uses bayes_xgb script
    params = xgb_maximise(X_train, y_train)

    print("Tuned Model Hyperparameters")
    for k, v in params.items():
        print("%s: %s" % (k, round(v, 2)))

    ##### Fitting model with tuned hyper-parameters
    tuned_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='aucpr')
    tuned_model.fit(X_train, y_train)

    # Making predictions on the held-out test data
    y_pred = tuned_model.predict(X_test)

    ##### Evaluation
    # Accuracy score
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc}')

    #AUPRC
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    print(f'precision score: {precision_score(y_test, y_pred)}')
    auc_precision_recall = auc(recall, precision)
    print(f'AUPRC: {auc_precision_recall}')

    #AUROC
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f'AUROC: {auc_roc}')

    #Sensitivity/Specificity
    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        return specificity

    def sensitivity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        return sensitivity
    print("Sensitivity: %s" % round(sensitivity_score(y_test, y_pred), 2))
    print("Specificity: %s" % round(specificity_score(y_test, y_pred), 2))

    # Plotting Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(12, 10))
    prc = plot_precision_recall_curve(tuned_model, X_test, y_test, ax=ax, **{"linewidth": 4})
    prc.ax_.set_title('Precision-Recall Curve of Sepsis Predictions: '
                      'AUC={0:0.2f}'.format(auc_precision_recall), pad=20, fontsize=18)
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.get_lines()[0].set_color(colours[2])
    ax.fill_between(prc.recall, prc.precision, interpolate=True, color="#F0EFB0", alpha=0.4)
    ax.get_legend().remove()
    ax.margins(x=0, y=0.01)
    fig.savefig(output_path + f"/AUPRC")

    # Plotting ROC curve
    fig, ax = plt.subplots(figsize=(12, 10))
    roc = plot_roc_curve(tuned_model, X_test, y_test, ax=ax, **{"linewidth": 4})
    roc.ax_.set_title('Receiver Operating Characteristic Curve of Sepsis Predictions: '
                      'AUC={0:0.2f}'.format(auc_roc), pad=20, fontsize=18)
    ax.get_legend().remove()
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.get_lines()[0].set_color(colours[2])
    plt.savefig(output_path + f"/AUROC")

    # Plotting Confusion Matrix
    # custom cmap
    norm = matplotlib.colors.Normalize(-1, 1)
    col_range = [[norm(-1.0), "white"],
              [norm(0), colours[0]],
              [norm(1.0), colours[1]]]
    ccmap = matplotlib.colors.LinearSegmentedColormap.from_list("", col_range)

    def plot_conf_mat(y_true, y_pred, col_map, name="Conf_mat"):
        cm = confusion_matrix(y_true, y_pred)
        norm_cm = np.array([x / np.sum(x) for x in cm])
        group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in norm_cm.flatten()]
        labels = [f"{v1}\n\n Count:{v2}\n\n Proportion:{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(norm_cm, annot=labels, fmt='', cmap=col_map, annot_kws={"fontsize":14},
                    xticklabels=["No Sepsis", "Sepsis"],
                    yticklabels=["No Sepsis", "Sepsis"])
        plt.yticks(rotation=0, fontsize=14)
        plt.xticks(rotation=0, fontsize=14)
        plt.ylabel('True Label', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=18, labelpad=20)
        fig.savefig(output_path + f"/{name}")

    plot_conf_mat(y_test, y_pred, ccmap)

    ##### Variable Importance
    # Retraining model with cluster assignments removed
    X_train_noclust = X_train.drop(["cluster_assignment"], axis=1)
    X_test_noclust = X_test.drop(["cluster_assignment"], axis=1)
    noclust_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='aucpr')
    noclust_model.fit(X_train_noclust, y_train)
    y_pred_noclust = noclust_model.predict(X_test_noclust)

    # Evaluating model with cluster assignments removed
    # AUPRC
    precision_noclust, recall_noclust, thresholds_noclust = precision_recall_curve(y_test, y_pred_noclust)
    auc_precision_recall_noclust = auc(recall_noclust, precision_noclust)
    print(f'AUPRC: {auc_precision_recall_noclust}')

    #AUROC
    auc_roc_noclust = roc_auc_score(y_test, y_pred_noclust)
    print(f'AUROC: {auc_roc_noclust}')

    # plotting variable importance
    fig, ax = plt.subplots(figsize=(12, 10))
    xgb.plot_importance(tuned_model, ax=ax, color=colours[4])
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_title("Feature Importance in XGboost Model", pad=20, fontsize=18)
    x_labels_formatted = ["Fibrinogen", "Resp. Rate", "WBC", "AST", "Sodium", "Gender", "ALT", "Cluster Assignment",
                "PTT", "Albumin", "Hemoglobin", "INR", "Age", "LDH", "SpO2", "Potassium", "Glucose", "CaO2",
                "Arterial pH", "Heart Rate", "NBP Mean", "CVP", "Temperature", "NBP (Diastolic)", "Urea",
                "NBP (Systolic)", "Creatinine", "Platelets"]
    ax.set_yticklabels(x_labels_formatted)
    fig.savefig(output_path + f"/Feature Importance")

if __name__ == "__main__":
    main()
