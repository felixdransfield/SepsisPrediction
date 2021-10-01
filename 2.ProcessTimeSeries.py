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
        1. Loads Sepsis time series data
        2. Cleans data & checks for outliers
        3. Aggregates time series into 1 value (using slope of regression if possible)
        4. Outputs dataset with 1 value for each variable
    """
    ###### Creating Directory for outputs
    output_path = "2.DATA CLEANING OUTPUT"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##### Data import
    # Reading csv data from the seperate Sepsis and No Sepsis Files
    sepsis = pd.read_csv("FinalSepsisSeries.csv")
    no_sepsis = pd.read_csv("FinalNonSepsisSeries.csv")
    # Creating outcome variable (sepsis = 1, no sepsis = 0)
    sepsis["diagnosis"] = 1
    no_sepsis["diagnosis"] = 0
    # Renaming Temperature variable so it is consistent
    sepsis.rename({"Temperature C (calc)":"Temperature C"}, axis=1, inplace=True)
    # Joining sepsis & no sepsis into 1 dataframe
    df = pd.concat([sepsis, no_sepsis], join='inner', ignore_index=True)

    # Based on domain specific knowledge the following features are dropped as not relevant to sepsis pathology
    # Chloride, Ionized Calcium, Magnesium
    df = df.drop(columns=["Chloride", "Ionized Calcium", "Magnesium"])

    ##### Data Cleaning
    # Plotting boxplots to visually inspect for outliers
    sns.set_style("whitegrid")
    bp_dims = (15, 8.27)
    fig, axes = plt.subplots(5, 6, figsize=bp_dims)
    fig.tight_layout(pad=2)
    axes = axes.flatten()
    for num, column in enumerate(df.iloc[:, 3:-3]):
        ax = sns.boxplot(y=column, x="diagnosis", data=df, palette="Set2", orient="v", width=0.6, ax=axes[num])
        ax.legend([], [], frameon=False)
        ax.set(xlabel=None, xticklabels=["No Sepsis", "Sepsis"])
    plt.savefig(output_path + "/TimeSeries variables boxplots", orientation='landscape')


    # Height has outlier/mistake (12 feet) will remove
    df['Admit Ht'].values[df['Admit Ht'] > 150] = np.nan
    # Arterial Ph as outlier/mistake (cant go higher than 14) will remove
    df['Arterial pH'].values[df['Arterial pH'] > 100] = np.nan


    # Everything else looks fine - redoing plot with outliers removed
    fig, axes = plt.subplots(5, 6, figsize=bp_dims)
    fig.tight_layout(pad=2)
    axes = axes.flatten()
    for num, column in enumerate(df.iloc[:, 3:-3]):
        ax = sns.boxplot(y=column, x="diagnosis", data=df, palette="Set2", orient="v", width=0.6, ax=axes[num])
        ax.legend([], [], frameon=False)
        ax.set(xlabel=None, xticklabels=["No Sepsis", "Sepsis"])
    plt.savefig(output_path + "/TimeSeries variables boxplots (outliers removed)", orientation='landscape')

    # Days have been duplicated in the data, removing duplicates and leaving only 24 hours
    def remove_duplicate_days(df):
        df_24 = df.iloc[:24, ]
        return df_24
    df = df.groupby(df["PatientID"]).apply(remove_duplicate_days)

    ##### Feature Engineering
    # Calculating BMI from Time Series data (Stored in separate BMI dataframe as a static feature)
    BMI = df[["PatientID", "Admit Ht", "Daily Weight"]].groupby(df["PatientID"]).max()
    BMI["Admit Ht"] = (BMI["Admit Ht"] * 2.54) / 100
    BMI["bmi"] = BMI["Daily Weight"] / (BMI["Admit Ht"] ** 2)
    # dropping height and weight from time series, will merge bmi dataframe when the time series is concatenated
    df = df.drop(columns=["Admit Ht", "Daily Weight"])
    BMI = BMI.drop(columns=["Admit Ht", "Daily Weight"])


    ##### Processing Time Series
    # Converting BUN to Urea
    df["BUN"] = df["BUN"] * 0.357
    df.rename({"BUN":"Urea"}, axis=1, inplace=True)

    # Based on domain specific knowledge select whether to use the highest, lowest or slope for the time series variables

    # Don't know = ["Arterial PaCO2", "Arterial PaO2", "SaO2"]

    # Features where the highest value is most relevant to sepsis
    highest = df[["PatientID", "ALT", "AST", "Urea", "Creatinine", "Fibrinogen", "INR", "LDH", "PTT", "Hemoglobin", "diagnosis"]]
    highest = highest.groupby(highest["PatientID"]).agg("max")
    # Features where the lowest value is most relevant to sepsis
    lowest = df[["PatientID", "Albumin", "NBP Mean", "NBP [Diastolic]", "NBP [Systolic]", "Platelets", "SpO2", "CaO2"]]
    lowest = lowest.groupby(lowest["PatientID"]).agg("min")
    # Features where the trend of values is most relevant to sepsis
    slopes = df[["PatientID", "Arterial pH", "CVP", "Glucose", "Heart Rate", "Potassium", "Resp Rate (Spont)", "Sodium", "WBC", "Temperature C", "OrdinalHour"]]
    def calculate_slopes(x):
        y = x["OrdinalHour"]
        for i in x.columns[1:-1]:
            colname = f"{i}_slope"
            if x[i].notnull().sum() > 1:
                mask = ~np.isnan(x[i]) & ~np.isnan(y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[i][mask], y[mask])
                x[colname] = slope
        return x

    slopes = slopes.groupby(slopes["PatientID"]).apply(calculate_slopes)
    slopes = slopes.iloc[:, 11:].groupby(slopes["PatientID"]).agg("max")

    # Merging together to form final DF
    df_aggregated = pd.merge(highest, lowest, left_on="PatientID", right_on="PatientID", how="left")
    df_aggregated = pd.merge(df_aggregated, slopes, left_on="PatientID", right_on="PatientID", how="left")


    ##### Adding variables from static data
    static_data = pd.read_csv("1.CLUSTERING OUTPUT\ClusteredDataDemographics.csv",
                              usecols=["PatientID", "gender", "AgeUnscaled", "cluster_assignment"])
    df_aggregated = pd.merge(df_aggregated, static_data, left_on="PatientID", right_on="PatientID", how="left")
    # BMI
    df_aggregated = pd.merge(df_aggregated, BMI, left_on="PatientID", right_on=BMI["PatientID"], how="left")
    # Final cleaning and removing of duplicates
    df_aggregated = df_aggregated.drop(columns=["PatientID_x", "PatientID_y"])
    df_aggregated = df_aggregated.drop_duplicates(subset=["PatientID"])


    ##### Output CSV
    df_aggregated.to_csv(output_path + "\cleaned_data.csv")


if __name__ == "__main__":
    main()