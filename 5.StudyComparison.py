# -*- coding: utf-8 -*-
##### Importing libraries
import os
import pandas as pd
import seaborn as sns


__author__ = "Felix Dransfield"


def main():
    """This script:
        plots comparison of AUROC, Sensitivity and Specificity for sepsis prediction studies.
    """

    ##### Creating Directory for outputs
    output_path = "5.STUDY COMPARISON OUTPUT"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ##### Data import
    # Reading csv data
    data = pd.read_csv("StudyComparison.csv")

    ##### Plotting
    g = sns.PairGrid(data,
                     x_vars=data.columns[1:-1], y_vars=["Paper"],
                     height=8, aspect=.25)
    g.map(sns.stripplot, size=10, orient="h", jitter=False,
          palette="flare_r", linewidth=1, edgecolor="w")
    for ax in g.axes.flat:

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        for (study, ticklbl) in zip(data['Study'], ax.yaxis.get_ticklabels()):
            ticklbl.set_color('red' if study == 1 else 'black')

    g.savefig(output_path + "/Study comparison", bbox_inches="tight")

if __name__ == "__main__":
    main()