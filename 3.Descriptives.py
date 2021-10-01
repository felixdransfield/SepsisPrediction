# -*- coding: utf-8 -*-
##### Importing libraries
import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Felix Dransfield"

def main():
    """This script:
        Generates descriptive statistics from the cleaned data output.
    """
    ###### Creating Directory for outputs
    output_path = "3.DESCRIPTIVES OUTPUT"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##### Data import
    # Reading csv data
    data = pd.read_csv("2.DATA CLEANING OUTPUT\cleaned_data.csv")

    ##### Descriptives

    # Summary statistics
    def summ_stats(df):
        print("Females: %s (%s)" % (df["gender"].value_counts()[0],
                                    round(df["gender"].value_counts(normalize=True)[0] * 100, 1)))
        print("Age: %s (%s)" % (round(df["AgeUnscaled"].describe()[1], 1), round(df["AgeUnscaled"].describe()[2], 1)))

    # total dataset
    summ_stats(data)

    # by diagnosis
    grouped = data.groupby("diagnosis")
    neg = grouped.get_group(0)
    pos = grouped.get_group(1)
    summ_stats(pos)
    summ_stats(neg)

    # cohort data
    sepsis_cohort = pd.read_csv("FinalSepsisCohort.csv")
    nosepsis_cohort = pd.read_csv("FinalNonSepsisCohort.csv")
    total_cohort = pd.concat([sepsis_cohort, nosepsis_cohort])

    def summ_cohort(df):
        print("Length of stay: %s (%s)" % (round(df["los"].describe()[1], 1), round(df["los"].describe()[2], 1)))
        print("Comorbidity: %s (%s)" % (round(df["comorbidity"].describe()[1], 1), round(df["comorbidity"].describe()[2], 1)))
        print("SOFA: %s (%s)" % (round(df["sofa"].describe()[1], 1), round(df["sofa"].describe()[2], 1)))
    summ_cohort(total_cohort)
    summ_cohort(sepsis_cohort)
    summ_cohort(nosepsis_cohort)

    # Descriptives by cluster
    clusters = data.groupby("cluster_assignment")
    for i in range(1, len(clusters)+1):
        print(f"cluster {i}:")
        cluster = clusters.get_group(i)
        print(f"n: {len(cluster)}")
        summ_stats(cluster)

    def summ_SOFA(df):
        print("cardiovascular: %s (%s)" % (round(df["cardiovascular"].describe()[1], 1), round(df["cardiovascular"].describe()[2], 1)))
        print("Respiratory: %s (%s)" % (round(df["respiration"].describe()[1], 1), round(df["respiration"].describe()[2], 1)))
        print("CNS: %s (%s)" % (round(df["cns"].describe()[1], 1), round(df["cns"].describe()[2], 1)))
        print("Liver: %s (%s)" % (round(df["liver"].describe()[1], 1), round(df["liver"].describe()[2], 1)))
        print("Renal: %s (%s)" % (round(df["renal"].describe()[1], 1), round(df["renal"].describe()[2], 1)))
        print("Coagulation: %s (%s)" % (round(df["coagulation"].describe()[1], 1), round(df["coagulation"].describe()[2], 1)))

    merged_clusters = pd.merge(data, total_cohort, left_on="PatientID", right_on="PatientID", how="left").groupby("cluster_assignment")
    for i in range(1, len(merged_clusters)+1):
        print(f"cluster {i}:")
        cluster = merged_clusters.get_group(i)
        summ_cohort(cluster)
        summ_SOFA(cluster)

    # Distribution of outcome
    def pct_and_values(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\nn = {:d}".format(pct, absolute)
    pie_data = data['diagnosis'].value_counts()
    plt.pie(pie_data, labels=['No Sepsis', 'Sepsis'], autopct=lambda pct: pct_and_values(pct, pie_data))
    plt.title("Distribution of Sepsis Outcome")
    plt.savefig(output_path + "/Distribution of Sepsis Outcome", orientation='landscape')

    # Distribution of features
    # By outcome
    feature_dict = {
        "maximum": ["ALT", "AST", "Urea", "Creatinine", "Fibrinogen", "INR", "LDH", "PTT", "Hemoglobin"],
        "minimum": ["Albumin", "NBP Mean", "NBP [Diastolic]", "NBP [Systolic]", "Platelets", "SpO2", "CaO2"],
        "trend": ["Arterial pH_slope", "CVP_slope", "Glucose_slope", "Heart Rate_slope", "Potassium_slope", "Resp Rate (Spont)_slope", "Sodium_slope", "WBC_slope", "Temperature C_slope"]
    }
    def FeatureKDE(df, var_list, name):
        grouped = df.groupby(df.diagnosis)
        neg = grouped.get_group(0)
        pos = grouped.get_group(1)
        sns.set_style("whitegrid")
        bp_dims = (15, 15)
        fig, axes = plt.subplots(3, 3, figsize=bp_dims)
        fig.tight_layout(pad=2)
        axes = axes.flatten()
        for num, feature in enumerate(var_list):
            t, p = stats.ttest_ind(neg[feature].dropna(), pos[feature].dropna(), equal_var=False)
            annotation = "t = %s\np = %s" % (round(t, 2), round(p, 3))
            if p < 0.05:
                weight = 'bold'
            else:
                weight = 'normal'
            ax = sns.kdeplot(neg[feature], shade=True, label="No Sepsis", bw_adjust=2, ax=axes[num])
            ax = sns.kdeplot(pos[feature], shade=True, label="Sepsis", bw_adjust=2, ax=axes[num])
            ax.legend()
            ax.annotate(annotation, (.8, .2), xycoords='axes fraction', backgroundcolor='w', weight=weight)
        plt.savefig(output_path + f"/Feature Distributions- {name}", orientation='landscape')

    for k, v in feature_dict.items():
        FeatureKDE(data, v, k)

    ##By cluster assignment

    def ClusterFeatureKDE(df, var_list, name):
        grouped = df.groupby(df.cluster_assignment)
        cardiogenic = grouped.get_group(1)
        respiratory = grouped.get_group(2)
        cns = grouped.get_group(3)
        renalLiver = grouped.get_group(4)
        sns.set_style("whitegrid")
        bp_dims = (15, 15)
        fig, axes = plt.subplots(3, 3, figsize=bp_dims)
        fig.tight_layout(pad=2)
        axes = axes.flatten()
        for num, feature in enumerate(var_list):
            F, p = stats.f_oneway(cardiogenic[feature].dropna(), respiratory[feature].dropna(),
                                  cns[feature].dropna(), renalLiver[feature].dropna())
            annotation = "F = %s\np = %s" % (round(F, 2), round(p, 3))
            if p < 0.05:
                weight = 'bold'
            else:
                weight = 'normal'
            ax = sns.kdeplot(cardiogenic[feature], shade=True, label="Cardiogenic", bw_adjust=2, ax=axes[num])
            ax = sns.kdeplot(respiratory[feature], shade=True, label="Respiratory", bw_adjust=2, ax=axes[num])
            ax = sns.kdeplot(cns[feature], shade=True, label="CNS", bw_adjust=2, ax=axes[num])
            ax = sns.kdeplot(renalLiver[feature], shade=True, label="Renal/Liver/Coagulation", bw_adjust=2, ax=axes[num])
            ax.legend()
            ax.annotate(annotation, (.8, .2), xycoords='axes fraction', backgroundcolor='w', weight=weight)
        plt.savefig(output_path + f"/Cluster Feature Distributions- {name}", orientation='landscape')

    for k, v in feature_dict.items():
        ClusterFeatureKDE(data, v, k)




if __name__ == "__main__":
    main()

