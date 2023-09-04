#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:05:30 2023

@author: troyobernolte

Meant to create visualizations for use in research paper
"""
import pandas as pd
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import data_reader
import os


plt.rcParams["font.family"] = "serif"

"""plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})"""

df = pd.read_csv("Data/clean_data.csv")
df_with_region = pd.read_csv("Data/data_with_region.csv")


def variable_skew(dataframe):
    # Compute skewness for each column
    skewness = dataframe.skew(axis=0)
    
    # Sort the skewness values in descending order
    sorted_skewness = skewness.sort_values(ascending=False)
    
    # Create 'Txt_files' directory if it doesn't exist
    os.makedirs("Txt_files", exist_ok=True)
    
    # Write skewness values to a text file in the 'Txt_files' directory
    with open("Txt_files/single_variable_skewness_values.txt", "w") as file:
        for column, skew_value in sorted_skewness.items():
            file.write(f"{column}: {skew_value}\n")
            
def sig_to_noise():
     x = [70, 160, 250, 350, 500]
     plt.xticks(x)
     for row in range(df.shape[0]):
         y = [df['Signi70'][row], df['Signi160'][row], df['Signi250'][row], 
              df['Signi350'][row], df['Signi500'][row]]
         if df['Coretype'][row] == 1:
             plt.plot(x,y, '^', color='purple', markersize = 5)
         elif df['Coretype'][row] == 2:
            plt.plot(x,y, 's', color='red', markersize = 5)
         elif df['Coretype'][row] == 3:
             plt.plot(x,y, '*', color='blue', markersize = 5)
     
     blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                          markersize=10, label='Protostellar')
     red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Prestellar')
     purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
                          markersize=10, label='Starless')
     
     plt.ylabel("Signal to Noise")
     plt.xlabel("Signal Level")
     plt.title("Signal to Noise")
     plt.legend(handles=[blue_star, red_square, purple_triangle], loc="upper right", title="Formation Stage")
     plt.savefig('Graphs/Signi.pdf')
     plt.show()
     
def sig_to_poiss():
     x = [70, 160, 250, 350, 500]
     plt.xticks(x)
     for row in range(df.shape[0]):
         y = [df['Sp70'][row], df['Sp160'][row], df['Sp250'][row], 
              df['Sp350'][row], df['Sp500'][row]]
         if df['Coretype'][row] == 1:
             plt.plot(x,y, '^', color='purple', markersize = 5)
         elif df['Coretype'][row] == 2:
            plt.plot(x,y, 's', color='red', markersize = 5)
         elif df['Coretype'][row] == 3:
             plt.plot(x,y, '*', color='blue', markersize = 5)
     
     blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                          markersize=10, label='Protostellar')
     red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Prestellar')
     purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
                          markersize=10, label='Starless')
     
     plt.ylabel("Signal to Poisson")
     plt.xlabel("Signal Level")
     plt.title("Signal to Poisson")
     plt.legend(handles=[blue_star, red_square, purple_triangle], loc="upper right", title="Formation Stage")
     plt.savefig('Graphs/Sp.pdf')
     plt.show()
     
def sp_sbg():
     x = [70, 160, 250, 350, 500]
     plt.xticks(x)
     for row in range(df.shape[0]):
         y = [df['Sp70/Sbg70'][row], df['Sp160/Sbg160'][row], df['Sp250/Sbg250'][row], 
              df['Sp350/Sbg350'][row], df['Sp500/Sbg500'][row]]
         if df['Coretype'][row] == 1:
             plt.plot(x,y, '^', color='purple', markersize = 5)
         elif df['Coretype'][row] == 2:
            plt.plot(x,y, 's', color='red', markersize = 5)
         elif df['Coretype'][row] == 3:
             plt.plot(x,y, '*', color='blue', markersize = 5)
     
     blue_star = mlines.Line2D([], [], color='blue', marker='*', linestyle='None',
                          markersize=10, label='Protostellar')
     red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                          markersize=10, label='Prestellar')
     purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
                          markersize=10, label='Starless')
     
     plt.ylabel("Sp/Sbg")
     plt.xlabel("Signal Level")
     plt.title("Sp/Sbg")
     plt.legend(handles=[blue_star, red_square, purple_triangle], loc="upper right", title="Formation Stage")
     plt.savefig('Graphs/Sp_sbg.pdf')
     plt.show()
     
def heatmap():
    corr = df.corr()
    heatmap = sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1)
    figure = heatmap.get_figure()
    figure.savefig("Graphs/heatmap.pdf", bbox_inches='tight', dpi=300)

def boxplot():
    for col in df.columns:
        if col != 'Region':
            plt.boxplot(df[col])
            plt.title("Boxplot of " + col)
            alias = col.replace('/', '_')
            save_path = 'Graphs/Boxes/box_{}.pdf'.format(alias)
            plt.savefig(save_path)
            plt.show()
            plt.clf()
    
def histogram():
    for col in df.columns:
        if col != 'Region':
            plt.hist(df[col], bins=4)
            plt.title("Histogram of " + col)
            alias = col.replace('/', '_')
            save_path = 'Graphs/Histograms/hist_{}.pdf'.format(alias)
            plt.savefig(save_path)
            plt.show()
            plt.clf()
            
            
def plot_hist_region(axis, region):
    region_data = df[df['Region'] == region]
    total = len(region_data)
    labels = {'Starless':(region_data['Coretype'] == 1).sum()/total,
              'Prestellar': (region_data['Coretype'] == 2).sum()/total,
              'Protostellar': (region_data['Coretype'] == 3).sum()/total}
    axis.bar(labels.keys(), labels.values())
    axis.set_title(region)
    axis.xaxis.set_tick_params(labelsize=7)
    
    
def bar_labels():
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_hist_region(axs[0,0], "Cepheus")
    plot_hist_region(axs[1,0], "Ophiuchus")
    plot_hist_region(axs[0,1], "Corona Australia")
    plot_hist_region(axs[1,1], "Lupus")
    save_path = 'Graphs/Bars/first_four.pdf'
    fig.savefig(save_path)

    

def labels_all():
    total = df.shape[0]
    labels = {'Starless':(df['Coretype'] == 1).sum(),
    'Prestellar': (df['Coretype'] == 2).sum(),
    'Protostellar': (df['Coretype'] == 3).sum()}
    plt.bar(labels.keys(), labels.values())
    bar_width = .1
    for i, key in enumerate(labels.keys()):
        x = 10*i * bar_width + bar_width / 2
        perc = round(labels[key]/total*100)
        plt.text(x, labels[key] + 15, '{} cores, or {}%'.format(labels[key], perc), ha='center')
    plt.title("Count of Coretypes")
    save_path = 'Graphs/Bars/Coretype.pdf'
    plt.savefig(save_path)
    plt.clf()

    
def labels_all_perc():
    total = df.shape[0]
    labels = {'Starless':(df['Coretype'] == 1).sum()/total*100,
    'Prestellar': (df['Coretype'] == 2).sum()/total*100,
    'Protostellar': (df['Coretype'] == 3).sum()/total*100}
    plt.bar(labels.keys(), labels.values())
    bar_width = .1
    for i, key in enumerate(labels.keys()):
        x = 10*i * bar_width + bar_width / 2
        plt.text(x, labels[key] + .07, '{}%'.format(round(labels[key])), ha='center')
    plt.title("Percentage of Coretypes")
    save_path = 'Graphs/Bars/Coretype_perc.pdf'
    plt.savefig(save_path)

    
    
    
class Region:
    """Class to simplify getting information on specific regions"""
    
    def __init__(self, region):
        self.name = region
        self.data = df_with_region.loc[df_with_region['Region'] == region, :]
        self.starless = self.data.loc[self.data['Coretype'] == 1].shape[0]
        self.prestellar = self.data.loc[self.data['Coretype'] == 2].shape[0]
        self.protostellar = self.data.loc[self.data['Coretype'] == 3].shape[0]

        
    def get_percs(self):
        total = self.data.shape[0]
        lst = []
        lst.append(self.starless / total)
        lst.append(self.prestellar / total)
        lst.append(self.protostellar / total)
        return lst

        

        

def bar_total_count():
    # set width of bars
    barWidth = 0.1
    #plt.figure().set_figwidth(15)
    regions = []
    for x in df['Region'].unique():
        regions.append(Region(x))
        
    # set heights of bars
    bars1 = [regions[0].data.shape[0]]
    bars2 = [regions[1].data.shape[0]]
    bars3 = [regions[2].data.shape[0]]
    bars4 = [regions[3].data.shape[0]]
    bars5 = [regions[4].data.shape[0]]
    bars6 = [regions[5].data.shape[0]]
    bars7 = [regions[6].data.shape[0]]
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = np.array([x + barWidth for x in r1])
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]
    
    # Make the plot
    plt.bar(r1, bars1, color='#FFA07A', width=barWidth, edgecolor='white', label='Cepheus')
    plt.bar(r2, bars2, color='#87CEFA', width=barWidth, edgecolor='white', label='Ophiuchus')
    plt.bar(r3, bars3, color='#FFDAB9', width=barWidth, edgecolor='white', label='Taurus')
    plt.bar(r4, bars4, color='#9370DB', width=barWidth, edgecolor='white', label='Corona Australis')
    plt.bar(r5, bars5, color='#98FB98', width=barWidth, edgecolor='white', label='Lupus')
    plt.bar(r6, bars6, color='#FFC0CB', width=barWidth, edgecolor='white', label='Aquilia')
    plt.bar(r7, bars7, color='#B0C4DE', width=barWidth, edgecolor='white', label='Orion')
    
    #plot text
    plt.text(r1, bars1[0] + 10, '{}'.format(bars1[0]), ha='center')
    plt.text(r2[0], bars2[0] + 10, '{}'.format(bars2[0]), ha='center')
    plt.text(r3[0], bars3[0] + 10, '{}'.format(bars3[0]), ha='center')
    plt.text(r4[0], bars4[0] + 10, '{}'.format(bars4[0]), ha='center')
    plt.text(r5[0], bars5[0] + 10, '{}'.format(bars5[0]), ha='center')
    plt.text(r6[0], bars6[0] + 10, '{}'.format(bars6[0]), ha='center')
    plt.text(r7[0], bars7[0] + 10, '{}'.format(bars7[0]), ha='center')

    
    # Add xticks on the middle of the group bars
    plt.xlabel('Region', fontweight='bold')
    plt.tick_params(labelbottom = False, bottom = False)
    
    plt.title("Count of Datapoints by region")

    # Create legend
    plt.legend()
    
    #Save file
    save_path = 'Graphs/datapoints_region.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    
def labels_grouped_bar(dataframe):
    # set width of bars
    barWidth = 0.1
    plt.figure().set_figwidth(15)
    regions = []
    for x in dataframe['Region'].unique():
        regions.append(Region(x))
        
    # set heights of bars
    bars1 = [regions[0].starless, regions[0].prestellar, regions[0].protostellar]
    bars2 = [regions[1].starless, regions[1].prestellar, regions[1].protostellar]
    bars3 = [regions[2].starless, regions[2].prestellar, regions[2].protostellar]
    bars4 = [regions[3].starless, regions[3].prestellar, regions[3].protostellar]
    bars5 = [regions[4].starless, regions[4].prestellar, regions[4].protostellar]
    bars6 = [regions[5].starless, regions[5].prestellar, regions[5].protostellar]
    bars7 = [regions[6].starless, regions[6].prestellar, regions[6].protostellar]
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1)).tolist()
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]


    
    # Make the plot
    plt.bar(r1, bars1, color='#FFA07A', width=barWidth, edgecolor='white', label='Cepheus')
    plt.bar(r2, bars2, color='#87CEFA', width=barWidth, edgecolor='white', label='Ophiuchus')
    plt.bar(r3, bars3, color='#FFDAB9', width=barWidth, edgecolor='white', label='Taurus')
    plt.bar(r4, bars4, color='#9370DB', width=barWidth, edgecolor='white', label='Corona Australis')
    plt.bar(r5, bars5, color='#98FB98', width=barWidth, edgecolor='white', label='Lupus')
    plt.bar(r6, bars6, color='#FFC0CB', width=barWidth, edgecolor='white', label='Aquilia')
    plt.bar(r7, bars7, color='#B0C4DE', width=barWidth, edgecolor='white', label='Orion')
    
    # Add the values on top of the bars
    for idx, bar_group in enumerate([r1, r2, r3, r4, r5, r6, r7]):
        for bar_idx, bar in enumerate(bar_group):
            height = [bars1, bars2, bars3, bars4, bars5, bars6, bars7][idx][bar_idx]
            plt.text(bar, height, f'{height}', ha='center', va='bottom', fontsize=8)

    
    # Add xticks on the middle of the group bars
    plt.xlabel('Coretype', fontweight='bold')
    plt.xticks([0.3,1.3,2.3], ['Starless','Prestellar', 'Protostellar'])
    
    plt.title("Count of coretypes by region")

    # Create legend
    plt.legend()
    
    #Save file
    save_path = 'Graphs/Coretype_bar_count.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    
def labels_grouped_bar_proportion(dataframe):
    # set width of bars
    barWidth = 0.1
    plt.figure().set_figwidth(15)
    regions = []
    for x in dataframe['Region'].unique():
        regions.append(Region(x))
        
    # set heights of bars
    bars1 = regions[0].get_percs()
    bars2 = regions[1].get_percs()
    bars3 = regions[2].get_percs()
    bars4 = regions[3].get_percs()
    bars5 = regions[4].get_percs()
    bars6 = regions[5].get_percs()
    bars7 = regions[6].get_percs()
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = np.array([x + barWidth for x in r1])
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]

    
    # Make the plot
    plt.bar(r1, bars1, color='#FFA07A', width=barWidth, edgecolor='white', label='Cepheus')
    plt.bar(r2, bars2, color='#87CEFA', width=barWidth, edgecolor='white', label='Ophiuchus')
    plt.bar(r3, bars3, color='#FFDAB9', width=barWidth, edgecolor='white', label='Taurus')
    plt.bar(r4, bars4, color='#9370DB', width=barWidth, edgecolor='white', label='Corona Australis')
    plt.bar(r5, bars5, color='#98FB98', width=barWidth, edgecolor='white', label='Lupus')
    plt.bar(r6, bars6, color='#FFC0CB', width=barWidth, edgecolor='white', label='Aquilia')
    plt.bar(r7, bars7, color='#B0C4DE', width=barWidth, edgecolor='white', label='Orion')
    
    # Add the values on top of the bars
    for idx, bar_group in enumerate([r1, r2, r3, r4, r5, r6, r7]):
        for bar_idx, bar in enumerate(bar_group):
            height = [bars1, bars2, bars3, bars4, bars5, bars6, bars7][idx][bar_idx]
            plt.text(bar, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Coretype', fontweight='bold')
    plt.xticks([0.3,1.3,2.3], ['Starless','Prestellar', 'Protostellar'])
    
    plt.title("Proportion of coretypes by region")

    # Create legend
    plt.legend()
    
    #Save file
    save_path = 'Graphs/Coretype_bar_proportion.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    
    
"""Unbalanced Results:
    
-------------------------------------------------------------------------------

"""
results_list_unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_unbalanced.append(["NB", 0.73, 0.73, 0.73, 0.73])
results_list_unbalanced.append(["Perceptron", 0.86, 0.85, 0.85, 0.85])
results_list_unbalanced.append(["SVM", 0.89, 0.89, 0.89, 0.89])
results_list_unbalanced.append(["SGD", 0.88, 0.87, 0.87, 0.87])
results_list_unbalanced.append(["Passive Aggressive", 0.84, 0.82, 0.81, 0.82])
results_list_unbalanced.append(["LDA", 0.87, 0.87, 0.87, 0.87])
results_list_unbalanced.append(["QDA", 0.82, 0.81, 0.81, 0.81])
results_list_unbalanced.append(["KNN", 0.86, 0.86, 0.86, 0.86])
results_list_unbalanced.append(["Decision Tree", 0.85, 0.84, 0.84, 0.84])
res_df_unbalanced = pd.DataFrame(results_list_unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])




"""Balanced Results:
    
-------------------------------------------------------------------------------

"""
results_list_balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_balanced.append(["NB", 0.75, 0.72, 0.73, 0.72])
results_list_balanced.append(["Perceptron", 0.83, 0.76, 0.77, 0.76])
results_list_balanced.append(["SVM", 0.89, 0.88, 0.88, 0.88])
results_list_balanced.append(["SGD", 0.87, 0.86, 0.86, 0.86])
results_list_balanced.append(["Passive Aggressive", 0.83, 0.80, 0.79, 0.80])
results_list_balanced.append(["LDA", 0.86, 0.86, 0.86, 0.86])
results_list_balanced.append(["QDA", 0.82, 0.81, 0.81, 0.81])
results_list_balanced.append(["KNN", 0.86, 0.86, 0.86, 0.86])
results_list_balanced.append(["Decision Tree", 0.80, 0.79, 0.79, 0.79])
res_df_balanced = pd.DataFrame(results_list_balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])
  



"""Unbalanced Results by region:
    
-------------------------------------------------------------------------------

"""

results_list_Cepheus_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Cepheus_Unbalanced.append(["NB", 0.76, 0.78, 0.75, 0.78])
results_list_Cepheus_Unbalanced.append(["Perceptron", 0.79, 0.78, 0.75, 0.78])
results_list_Cepheus_Unbalanced.append(["SVM", 0.81, 0.79, 0.76, 0.79])
results_list_Cepheus_Unbalanced.append(["SGD", 0.82, 0.79, 0.78, 0.79])
results_list_Cepheus_Unbalanced.append(["Passive Aggressive", 0.8, 0.8, 0.79, 0.8])
results_list_Cepheus_Unbalanced.append(["LDA", 0.78, 0.77, 0.75, 0.77])
results_list_Cepheus_Unbalanced.append(["QDA", 0.77, 0.61, 0.67, 0.61])
results_list_Cepheus_Unbalanced.append(["KNN", 0.68, 0.73, 0.67, 0.73])
results_list_Cepheus_Unbalanced.append(["Decision Tree", 0.8, 0.79, 0.78, 0.79])
res_df_Cepheus_Unbalanced = pd.DataFrame(results_list_Cepheus_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Ophiuchus_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Ophiuchus_Unbalanced.append(["NB", 0.57, 0.71, 0.61, 0.71])
results_list_Ophiuchus_Unbalanced.append(["Perceptron", 0.86, 0.85, 0.84, 0.85])
results_list_Ophiuchus_Unbalanced.append(["SVM", 0.87, 0.84, 0.83, 0.84])
results_list_Ophiuchus_Unbalanced.append(["SGD", 0.84, 0.83, 0.82, 0.83])
results_list_Ophiuchus_Unbalanced.append(["Passive Aggressive", 0.83, 0.83, 0.81, 0.83])
results_list_Ophiuchus_Unbalanced.append(["LDA", 0.66, 0.72, 0.66, 0.72])
results_list_Ophiuchus_Unbalanced.append(["QDA", 0.83, 0.83, 0.80, 0.83])
results_list_Ophiuchus_Unbalanced.append(["KNN", 0.80, 0.80, 0.76, 0.80])
results_list_Ophiuchus_Unbalanced.append(["Decision Tree", 0.88, 0.85, 0.85, 0.85])
res_df_Ophiuchus_Unbalanced = pd.DataFrame(results_list_Ophiuchus_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])



results_list_Taurus_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Taurus_Unbalanced.append(["NB", 0.87, 0.86, 0.86, 0.86])
results_list_Taurus_Unbalanced.append(["Perceptron", 0.89, 0.88, 0.88, 0.88])
results_list_Taurus_Unbalanced.append(["SVM", 0.88, 0.92, 0.89, 0.92])
results_list_Taurus_Unbalanced.append(["SGD", 0.85, 0.84, 0.83, 0.84])
results_list_Taurus_Unbalanced.append(["Passive Aggressive", 0.88, 0.91, 0.89, 0.91])
results_list_Taurus_Unbalanced.append(["LDA", 0.93, 0.96, 0.94, 0.96])
results_list_Taurus_Unbalanced.append(["QDA", 0.86, 0.86, 0.86, 0.86])
results_list_Taurus_Unbalanced.append(["KNN", 0.78, 0.88, 0.83, 0.88])
results_list_Taurus_Unbalanced.append(["Decision Tree", 0.87, 0.9, 0.88, 0.9])
res_df_Taurus_Unbalanced = pd.DataFrame(results_list_Taurus_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Corona_Australis_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Corona_Australis_Unbalanced.append(["NB", 0.68, 0.79, 0.73, 0.79])
results_list_Corona_Australis_Unbalanced.append(["Perceptron", 0.72, 0.78, 0.74, 0.78])
results_list_Corona_Australis_Unbalanced.append(["SVM", 0.61, 0.76, 0.67, 0.76])
results_list_Corona_Australis_Unbalanced.append(["SGD", 0.73, 0.78, 0.74, 0.78])
results_list_Corona_Australis_Unbalanced.append(["Passive Aggressive", 0.83, 0.88, 0.85, 0.88])
results_list_Corona_Australis_Unbalanced.append(["LDA", 0.26, 0.29, 0.26, 0.29])
results_list_Corona_Australis_Unbalanced.append(["QDA", 0.09, 0.18, 0.11, 0.18])
results_list_Corona_Australis_Unbalanced.append(["KNN", 0.54, 0.72, 0.61, 0.72])
results_list_Corona_Australis_Unbalanced.append(["Decision Tree", 0.84, 0.88, 0.86, 0.88])
res_df_Corona_Australis_Unbalanced = pd.DataFrame(results_list_Corona_Australis_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Lupus_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Lupus_Unbalanced.append(["NB", 0.77, 0.74, 0.74, 0.74])
results_list_Lupus_Unbalanced.append(["Perceptron", 0.8, 0.8, 0.78, 0.8])
results_list_Lupus_Unbalanced.append(["SVM", 0.67, 0.78, 0.7, 0.78])
results_list_Lupus_Unbalanced.append(["SGD", 0.7, 0.67, 0.65, 0.67])
results_list_Lupus_Unbalanced.append(["Passive Aggressive", 0.8, 0.78, 0.77, 0.78])
results_list_Lupus_Unbalanced.append(["LDA", 0.83, 0.83, 0.81, 0.83])
results_list_Lupus_Unbalanced.append(["QDA", 0.79, 0.83, 0.79, 0.83])
results_list_Lupus_Unbalanced.append(["KNN", 0.58, 0.76, 0.66, 0.76])
results_list_Lupus_Unbalanced.append(["Decision Tree", 0.83, 0.82, 0.81, 0.82])
res_df_Lupus_Unbalanced = pd.DataFrame(results_list_Lupus_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Aquilia_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Aquilia_Unbalanced.append(["NB", 0.75, 0.51, 0.46, 0.51])
results_list_Aquilia_Unbalanced.append(["Perceptron", 0.65, 0.63, 0.58, 0.63])
results_list_Aquilia_Unbalanced.append(["SVM", 0.45, 0.59, 0.47, 0.59])
results_list_Aquilia_Unbalanced.append(["SGD", 0.54, 0.6, 0.52, 0.6])
results_list_Aquilia_Unbalanced.append(["Passive Aggressive", 0.66, 0.61, 0.58, 0.61])
results_list_Aquilia_Unbalanced.append(["LDA", 0.74, 0.72, 0.71, 0.72])
results_list_Aquilia_Unbalanced.append(["QDA", 0.75, 0.72, 0.71, 0.72])
results_list_Aquilia_Unbalanced.append(["KNN", 0.53, 0.6, 0.54, 0.6])
results_list_Aquilia_Unbalanced.append(["Decision Tree", 0.74, 0.74, 0.73, 0.74])
res_df_Aquilia_Unbalanced = pd.DataFrame(results_list_Aquilia_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Orion_Unbalanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Orion_Unbalanced.append(["NB", 0.75, 0.68, 0.62, 0.68])
results_list_Orion_Unbalanced.append(["Perceptron", 0.68, 0.69, 0.65, 0.69])
results_list_Orion_Unbalanced.append(["SVM", 0.78, 0.75, 0.71, 0.75])
results_list_Orion_Unbalanced.append(["SGD", 0.75, 0.7, 0.67, 0.7])
results_list_Orion_Unbalanced.append(["Passive Aggressive", 0.76, 0.69, 0.66, 0.69])
results_list_Orion_Unbalanced.append(["LDA", 0.76, 0.76, 0.74, 0.76])
results_list_Orion_Unbalanced.append(["QDA", 0.76, 0.73, 0.71, 0.73])
results_list_Orion_Unbalanced.append(["KNN", 0.73, 0.72, 0.67, 0.72])
results_list_Orion_Unbalanced.append(["Decision Tree", 0.84, 0.84, 0.84, 0.84])
res_df_Orion_Unbalanced = pd.DataFrame(results_list_Orion_Unbalanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])



"""Balanced Results by region:
    
-------------------------------------------------------------------------------

"""
results_list_Cepheus_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Cepheus_Balanced.append(["NB", 0.45, 0.61, 0.47, 0.61])
results_list_Cepheus_Balanced.append(["Perceptron", 0.76, 0.71, 0.7, 0.71])
results_list_Cepheus_Balanced.append(["SVM", 0.81, 0.8, 0.78, 0.8])
results_list_Cepheus_Balanced.append(["SGD", 0.81, 0.77, 0.75, 0.77])
results_list_Cepheus_Balanced.append(["Passive Aggressive", 0.79, 0.79, 0.77, 0.79])
results_list_Cepheus_Balanced.append(["LDA", 0.82, 0.8, 0.77, 0.8])
results_list_Cepheus_Balanced.append(["QDA", 0.77, 0.61, 0.67, 0.61])
results_list_Cepheus_Balanced.append(["KNN", 0.72, 0.75, 0.72, 0.75])
results_list_Cepheus_Balanced.append(["Decision Tree", 0.75, 0.72, 0.72, 0.72])
res_df_Cepheus_Balanced = pd.DataFrame(results_list_Cepheus_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])



results_list_Ophiuchus_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Ophiuchus_Balanced.append(["NB", 0.57, 0.71, 0.61, 0.71])
results_list_Ophiuchus_Balanced.append(["Perceptron", 0.86, 0.85, 0.84, 0.85])
results_list_Ophiuchus_Balanced.append(["SVM", 0.87, 0.84, 0.83, 0.84])
results_list_Ophiuchus_Balanced.append(["SGD", 0.83, 0.75, 0.75, 0.75])
results_list_Ophiuchus_Balanced.append(["Passive Aggressive", 0.85, 0.85, 0.83, 0.85])
results_list_Ophiuchus_Balanced.append(["LDA", 0.66, 0.72, 0.66, 0.72])
results_list_Ophiuchus_Balanced.append(["QDA", 0.83, 0.83, 0.80, 0.83])
results_list_Ophiuchus_Balanced.append(["KNN", 0.80, 0.80, 0.76, 0.80])
results_list_Ophiuchus_Balanced.append(["Decision Tree", 0.86, 0.83, 0.83, 0.83])
res_df_Ophiuchus_Balanced = pd.DataFrame(results_list_Ophiuchus_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Taurus_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Taurus_Balanced.append(["NB", 0.78, 0.88, 0.83, 0.88])
results_list_Taurus_Balanced.append(["Perceptron", 0.92, 0.83, 0.85, 0.83])
results_list_Taurus_Balanced.append(["SVM", 0.9, 0.88, 0.88, 0.88])
results_list_Taurus_Balanced.append(["SGD", 0.73, 0.67, 0.67, 0.67])
results_list_Taurus_Balanced.append(["Passive Aggressive", 0.9, 0.89, 0.89, 0.89])
results_list_Taurus_Balanced.append(["LDA", 0.87, 0.92, 0.89, 0.92])
results_list_Taurus_Balanced.append(["QDA", 0.86, 0.86, 0.86, 0.86])
results_list_Taurus_Balanced.append(["KNN", 0.82, 0.9, 0.85, 0.9])
results_list_Taurus_Balanced.append(["Decision Tree", 0.9, 0.88, 0.88, 0.88])
res_df_Taurus_Balanced = pd.DataFrame(results_list_Taurus_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Corona_Australis_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Corona_Australis_Balanced.append(["NB", 0.65, 0.76, 0.69, 0.76])
results_list_Corona_Australis_Balanced.append(["Perceptron", 0.74, 0.75, 0.74, 0.75])
results_list_Corona_Australis_Balanced.append(["SVM", 0.63, 0.76, 0.68, 0.76])
results_list_Corona_Australis_Balanced.append(["SGD", 0.68, 0.76, 0.71, 0.76])
results_list_Corona_Australis_Balanced.append(["Passive Aggressive", 0.83, 0.88, 0.85, 0.88])
results_list_Corona_Australis_Balanced.append(["LDA", 0.71, 0.79, 0.74, 0.79])
results_list_Corona_Australis_Balanced.append(["QDA", 0.09, 0.18, 0.11, 0.18])
results_list_Corona_Australis_Balanced.append(["KNN", 0.62, 0.72, 0.66, 0.72])
results_list_Corona_Australis_Balanced.append(["Decision Tree", 0.94, 0.91, 0.92, 0.91])
res_df_Corona_Australis_Balanced = pd.DataFrame(results_list_Corona_Australis_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])

results_list_Lupus_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Lupus_Balanced.append(["NB", 0.58, 0.75, 0.65, 0.75])
results_list_Lupus_Balanced.append(["Perceptron", 0.81, 0.78, 0.77, 0.78])
results_list_Lupus_Balanced.append(["SVM", 0.81, 0.78, 0.78, 0.78])
results_list_Lupus_Balanced.append(["SGD", 0.78, 0.78, 0.76, 0.78])
results_list_Lupus_Balanced.append(["Passive Aggressive", 0.78, 0.83, 0.79, 0.83])
results_list_Lupus_Balanced.append(["LDA", 0.78, 0.84, 0.79, 0.84])
results_list_Lupus_Balanced.append(["QDA", 0.79, 0.83, 0.79, 0.83])
results_list_Lupus_Balanced.append(["KNN", 0.75, 0.81, 0.76, 0.81])
results_list_Lupus_Balanced.append(["Decision Tree", 0.84, 0.74, 0.75, 0.74])
res_df_Lupus_Balanced = pd.DataFrame(results_list_Lupus_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Aquilia_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Aquilia_Balanced.append(["NB", 0.23, 0.34, 0.2, 0.34])
results_list_Aquilia_Balanced.append(["Perceptron", 0.6, 0.6, 0.55, 0.6])
results_list_Aquilia_Balanced.append(["SVM", 0.68, 0.54, 0.51, 0.54])
results_list_Aquilia_Balanced.append(["SGD", 0.58, 0.53, 0.5, 0.53])
results_list_Aquilia_Balanced.append(["Passive Aggressive", 0.61, 0.62, 0.56, 0.62])
results_list_Aquilia_Balanced.append(["LDA", 0.58, 0.61, 0.57, 0.61])
results_list_Aquilia_Balanced.append(["QDA", 0.75, 0.72, 0.71, 0.72])
results_list_Aquilia_Balanced.append(["KNN", 0.56, 0.57, 0.55, 0.57])
results_list_Aquilia_Balanced.append(["Decision Tree", 0.76, 0.67, 0.66, 0.67])
res_df_Aquilia_Balanced = pd.DataFrame(results_list_Aquilia_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


results_list_Orion_Balanced = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list_Orion_Balanced.append(["NB", 0.53, 0.58, 0.45, 0.58])
results_list_Orion_Balanced.append(["Perceptron", 0.77, 0.74, 0.73, 0.74])
results_list_Orion_Balanced.append(["SVM", 0.79, 0.75, 0.73, 0.75])
results_list_Orion_Balanced.append(["SGD", 0.73, 0.66, 0.65, 0.66])
results_list_Orion_Balanced.append(["Passive Aggressive", 0.75, 0.69, 0.65, 0.69])
results_list_Orion_Balanced.append(["LDA", 0.67, 0.68, 0.63, 0.68])
results_list_Orion_Balanced.append(["QDA", 0.76, 0.73, 0.71, 0.73])
results_list_Orion_Balanced.append(["KNN", 0.69, 0.70, 0.67, 0.70])
results_list_Orion_Balanced.append(["Decision Tree", 0.81, 0.79, 0.79, 0.79])
res_df_Orion_Balanced = pd.DataFrame(results_list_Orion_Balanced, columns=['Name', 'Precision', 'Recall', 'F1', 'Accuracy'])


colors = {'NB': 'blue', 'Perceptron':'darkturquoise', 'SVM':'darkorange', 'SGD':'red',
          'Passive Aggressive':'purple', 'LDA':'brown', 'QDA':'pink',
          'KNN':'gray', 'Decision Tree':'green'}


def chart_precision(BALANCE,name):
    print("I am working with:", BALANCE)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dummy = BALANCE.sort_values(by=['Precision'], ascending=False)
    bars = ax.bar(x=dummy['Name'], height=dummy['Precision'],
                   width=0.3, color=['blue', 'green', 'orange', 'red', 'purple',
                                      'brown','pink','gray','cyan'])
    plt.tick_params(axis='x', labelsize=25)
    
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.ylim(0.6, 0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
          bar.get_height(),
          horizontalalignment='center',
          weight='bold'
          )
    plt.tick_params(axis='x', labelbottom = False, bottom=False)
    title = "{} Model Precision With K=10".format(name)
    if name == "Balanced":
        title = "{} Model Precision".format(name)
    plt.title(title, fontsize=18)    
    c = dummy['Name'].apply(lambda x: colors[x])
    ax = plt.subplot(111) #specify a subplot

    bars = ax.bar(dummy['Name'], dummy['Precision'], color=c) #Plot data on subplot axis

    for i, j in colors.items(): #Loop over color dictionary
        ax.bar(dummy['Name'], dummy['Precision'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc=1, prop={'size': 6})
    
    plt.savefig("Graphs/Results/Precision_{}.pdf".format(name), bbox_inches='tight')
    return fig

def chart_recall(BALANCE, name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dummy = BALANCE.sort_values(by=['Recall'], ascending=False)
    bars = ax.bar(x=dummy['Name'], height=dummy['Recall'],
                   width=0.3, color=['blue', 'green', 'orange', 'red', 'purple',
                                      'brown','pink','gray','cyan'])
    plt.tick_params(axis='x', labelsize=25)
    
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.ylim(0.6, 0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
          bar.get_height(),
          horizontalalignment='center',
          weight='bold'
          )
    plt.tick_params(axis='x', labelbottom = False, bottom=False)
    
    c = dummy['Name'].apply(lambda x: colors[x])
    ax = plt.subplot(111) #specify a subplot

    bars = ax.bar(dummy['Name'], dummy['Recall'], color=c) #Plot data on subplot axis

    for i, j in colors.items(): #Loop over color dictionary
        ax.bar(dummy['Name'], dummy['Recall'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc=1, prop={'size': 6})
    
    title = "{} Model Recall With K=10".format(name)
    plt.title(title, fontsize=18)
    plt.savefig("Graphs/Results/Recall_{}.pdf".format(name), bbox_inches='tight')
    return plt
    
def chart_f1(BALANCE,name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dummy = BALANCE.sort_values(by=['F1'], ascending=False)
    bars = ax.bar(x=dummy['Name'], height=dummy['F1'],
                   width=0.3, color=['blue', 'green', 'orange', 'red', 'purple',
                                      'brown','pink','gray','cyan'])
    plt.tick_params(axis='x', labelsize=25)
    
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.ylim(0.6, 0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
          bar.get_height(),
          horizontalalignment='center',
          weight='bold'
          )
    plt.tick_params(axis='x', labelbottom = False, bottom=False)
    
    c = dummy['Name'].apply(lambda x: colors[x])
    ax = plt.subplot(111) #specify a subplot

    bars = ax.bar(dummy['Name'], dummy['F1'], color=c) #Plot data on subplot axis

    for i, j in colors.items(): #Loop over color dictionary
        ax.bar(dummy['Name'], dummy['F1'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc=1, prop={'size': 6})
    title = "{} Model F1 With K=10".format(name)
    plt.title(title, fontsize=18)
    plt.savefig("Graphs/Results/F1_{}.pdf".format(name), bbox_inches='tight')
    return plt

def chart_accuracy(BALANCE,name):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dummy = BALANCE.sort_values(by=['Accuracy'], ascending=False)
    bars = ax.bar(x=dummy['Name'], height=dummy['Accuracy'],
                   width=0.3, color=['blue', 'green', 'orange', 'red', 'purple',
                                      'brown','pink','gray','cyan'])
    plt.tick_params(axis='x', labelsize=25)
    
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    plt.ylim(0.6, 0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
          round(bar.get_height(),3),
          horizontalalignment='center',
          weight='bold'
          )
    plt.tick_params(axis='x', labelbottom = False, bottom=False)
    
    c = dummy['Name'].apply(lambda x: colors[x])
    ax = plt.subplot(111) #specify a subplot

    bars = ax.bar(dummy['Name'], dummy['Accuracy'], color=c) #Plot data on subplot axis

    for i, j in colors.items(): #Loop over color dictionary
        ax.bar(dummy['Name'], dummy['Accuracy'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc=1, prop={'size': 6})
    title = "{} Model Accuracy With K=10".format(name)
    plt.title(title, fontsize=18)
    plt.savefig("Graphs/Results/Accuracy_{}.pdf".format(name), bbox_inches='tight')
    return plt

def graph_bars():
    figure, axis = plt.subplots(2, 2)
    axis[0,0].chart_accuracy()

            
def pairplot():
    
    # Drop columns with string values
    df_numeric = df.select_dtypes(include="number")      
    # Take log of data
    df_log = pd.concat([df['Region'], np.log10(df_numeric)], axis=1)
        
    for cat in df_numeric.columns:
        data_pos = df_log[df_log[cat] > 0]
        box = sns.boxplot(data=data_pos, y="Region", x=cat)
        plt.title("Boxplot of log {} by Region".format(cat))
        plt.show()
        fig = box.get_figure()
        save_path = "Graphs/Pairplots/{}.pdf".format(cat.replace("/", "_"))
        fig.savefig(save_path)
                
def cor():
    testing_length = '70'
    cor = df.corr()
    wavelengths = ['70', '160', '250', '350', '500']
    cats = ['Sp', 'Sconv', 'Stot', 'FWHMa', 'FWHMb']
    for var in cats:
        for num in wavelengths:
            if num != testing_length:
                x = var + testing_length
                var2 = var+num
                if x == "FWHMb70":
                    x = "FWHMb070"
                if var2 == 'FWHMb70':
                    var2 = 'FWHMb070'
                if var2 != 'Sconv500':
                    correlation = round(cor[x][var2], 3)
                    print("Correlation between {} and {}: {}".format(x, var2, correlation))
                    scatter(x, var2)
        
def scatter(var1, var2):
    x = df[var1]
    y = df[var2]
    plt.scatter(x, y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    a, b = np.polyfit(x, y, 1)
    plt.plot(x, slope*x+intercept, color='red', linestyle='--', linewidth=2)      
    plt.text(0.1, 0.9, 'y = ' + '{:.2f}'.format(intercept) + 
             ' + {:.2f}'.format(slope) + 'x', transform=plt.gca().transAxes, 
             size=14)
    
    r_squared = r_value**2
    plt.text(0.1, 0.8, f'R² = {r_squared:.2f}', transform=plt.gca().transAxes, 
             size = 14)

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(var1 + "&" + var2)
    savepath = "Graphs/Scatter/{}_{}".format(var1, var2)
    plt.savefig(savepath)
    plt.show()

def plot_nans():
    df = data_reader.main()
    #These are the only dropped columns that contain NaNs
    df = df.drop(['Signi70', 'Signi500'], axis=1)
    data_clean(df)

    # Count the number of NaN values in each column
    nan_count = df.isna().sum()

    # Filter out columns with no NaN values
    nan_count = nan_count[nan_count > 0]

    # Convert the Series to a DataFrame
    nan_count_df = nan_count.reset_index()
    nan_count_df.columns = ['Column', 'NaN Count']

    # Sort the DataFrame by the number of NaN values
    nan_count_df = nan_count_df.sort_values(by=['NaN Count'], ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(nan_count_df['Column'], nan_count_df['NaN Count'])
    plt.xlabel('Columns')
    plt.ylabel('Number of NaN Values')
    plt.xticks(rotation=45)
    plt.title('Bar Chart of NaN Values in Each Column')
    savepath = "Graphs/nans.pdf"
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

    
def data_clean(df):
    """Cleans data. Should be run before anything else"""
    for col in df.select_dtypes(include='object').columns:
        for row in range(df.shape[0]):
            if not(col == "Region" or col == "Coretpye"):
                try:
                    df[col][row] = df[col][row].str.strip()
                except AttributeError:
                    try: 
                        df[col][row] = float(df[col][row])
                    except AttributeError:
                        df[col][row] = np.nan
                    except ValueError:
                        df[col][row] = np.nan
    if (col == "Region"):
        df[col] = df[col].astype(object)
    elif (col == "NSED" or col == "Coretype"):
        df[col] = df[col].astype(int)
    else:
        df[col] =df[col].astype(float) 

def plot_balanced_data_region(balance):
    # Data
    regions = ['Cepheus', 'Ophiuchus', 'Taurus', 'Corona Australis', 'Lupus', 'Aquilia', 'Orion']
    if balance == "Unbalanced":
        counts = [587, 345, 344, 68, 381, 526, 1323]
    elif balance == "Balanced":
        counts = [84, 74, 41, 14, 73, 98, 216]
    total_count = sum(counts)
    proportions = [count / total_count for count in counts]
    
    # Barplot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    bar_width = 0.6
    bar_spacing = 0.7
    x_positions = [i * bar_spacing for i in range(len(regions))]
    bars = ax.bar(x_positions, counts, width=bar_width)
    
    # Add count and proportion labels
    for i, bar in enumerate(bars):
        count = counts[i]
        proportion = proportions[i]
        label = f"{count} ({proportion:.2%})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom')
    
    # Customize plot
    ax.set_ylabel('Count')
    ax.set_title(f'Data by Region Contributed To {balance} Training Dataset', fontsize=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"Graphs/Bars/{balance}_training_data.pdf", bbox_inches='tight', dpi=300)
    
    # Show plot
    plt.show()

    
def chart(df, name):
    chart_precision(df, name)
    chart_recall(df, name)
    chart_f1(df, name)
    chart_accuracy(df, name)
    
def plot_metric(metric, title, balance):
    save_path = f"Graphs/Results/Regions/{balance}"
    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    index = np.arange(7)
    bar_width = 0.25
    opacity = 0.8
    
    colors = {'NB': 'blue', 'Perceptron': 'darkturquoise', 'SVM': 'darkorange', 'SGD': 'red',
              'Passive Aggressive': 'purple', 'LDA': 'brown', 'QDA': 'pink',
              'KNN': 'gray', 'Decision Tree': 'green'}
    if balance == "Balanced":
        bar_groups = [
        ("Cepheus", res_df_Cepheus_Balanced[metric].nlargest(3)),
        ("Ophiuchus", res_df_Ophiuchus_Balanced[metric].nlargest(3)),
        ("Taurus", res_df_Taurus_Balanced[metric].nlargest(3)),
        ("Corona Australis", res_df_Corona_Australis_Balanced[metric].nlargest(3)),
        ("Lupus", res_df_Lupus_Balanced[metric].nlargest(3)),
        ("Aquilia", res_df_Aquilia_Balanced[metric].nlargest(3)),
        ("Orion", res_df_Orion_Balanced[metric].nlargest(3)),
        ]
    else:
        bar_groups = [
        ("Cepheus", res_df_Cepheus_Unbalanced[metric].nlargest(3)),
        ("Ophiuchus", res_df_Ophiuchus_Unbalanced[metric].nlargest(3)),
        ("Taurus", res_df_Taurus_Unbalanced[metric].nlargest(3)),
        ("Corona Australis", res_df_Corona_Australis_Unbalanced[metric].nlargest(3)),
        ("Lupus", res_df_Lupus_Unbalanced[metric].nlargest(3)),
        ("Aquilia", res_df_Aquilia_Unbalanced[metric].nlargest(3)),
        ("Orion", res_df_Orion_Unbalanced[metric].nlargest(3)),
        ]
    
    region_names = [
        "Cepheus",
        "Ophiuchus",
        "Taurus",
        "Corona Australis",
        "Lupus",
        "Aquilia",
        "Orion",
    ]
    
    seen = set()
    
    for bar_offset, (region_name, top_performers) in enumerate(bar_groups):
        for i, (idx, score) in enumerate(top_performers.iteritems()):
            top_performing_method = res_df_Cepheus_Unbalanced.loc[idx]['Name']
            color = colors[top_performing_method]
            
            if top_performing_method not in seen:
                label = top_performing_method
                seen.add(top_performing_method)
            else:
                label = ""
            
            bar = plt.bar(index[bar_offset] + i * bar_width, score, bar_width, alpha=opacity, color=color, label=label)
            
            # Add the numeric value above each bar
            ax.annotate(
                f"{score:.2f}",
                xy=(bar[0].get_x() + bar_width / 2, score),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,  # Adjust the font size here
                fontweight='bold'  # Make the text bold
            )
    
    
    plt.xlabel('Region')
    plt.ylabel(metric)
    plt.title(title, fontsize=20)  # Increase the title fontsize
    plt.xticks(index + bar_width * 1.5, region_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Position the legend outside the main figure
    plt.tight_layout()
    
    # Save the figure as a PDF file
    plt.savefig(os.path.join(save_path, f"{title}.pdf"), format='pdf', dpi=300)
    
    plt.show()


def SED_Analysis(df, y_max=26):
    wavelengths = [70, 160, 250, 350, 500]
    stot_columns = ['Stot70', 'Stot160', 'Stot250', 'Stot350', 'Stot500']
    coretypes = [1, 2, 3]
    coretype_names = {1: 'Starless', 2: 'Prestellar', 3: 'Protostellar'}
    
    # Loop through each coretype
    for coretype in coretypes:
        # Filter the dataframe for the current coretype
        coretype_df = df[df['Coretype'] == coretype]

        # Calculate the average Stot values for each wavelength
        avg_stot_values = coretype_df[stot_columns].mean().values

        # Create a figure and axis for the current plot
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        # Plot the average Stot values for the current coretype
        ax.plot(wavelengths, avg_stot_values, marker='o', linestyle='-')
        ax.set_title(f'Spectral Energy Distribution (SED)\n of {coretype_names[coretype]} Coretype', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Wavelength (µm)')
        ax.set_ylabel('Average Stot')

        # Set the x-axis scale to log, display only specified wavelength ticks, and avoid scientific notation
        ax.set_xscale('log')
        ax.set_xticks(wavelengths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # Set the y-axis scale to linear, standardize the y-axis range, and avoid scientific notation
        ax.set_ylim(0, y_max)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

        # Save the figure as a high-quality PDF
        plt.tight_layout()
        plt.savefig(f'Sed Analysis/SED_Analysis_{coretype_names[coretype]}_Coretype.pdf', format='pdf', dpi=300)
        plt.show()
        plt.close(fig)

def SED_analysis_combined(df, y_max=26):
    wavelengths = [70, 160, 250, 350, 500]
    stot_columns = ['Stot70', 'Stot160', 'Stot250', 'Stot350', 'Stot500']
    coretypes = [1, 2, 3]
    coretype_names = {1: 'Starless', 2: 'Prestellar', 3: 'Protostellar'}
    
    # Create a figure with three subplots (one for each coretype)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=300)
    
    # Loop through each coretype
    for index, coretype in enumerate(coretypes):
        ax = axes[index]
        
        # Filter the dataframe for the current coretype
        coretype_df = df[df['Coretype'] == coretype]

        # Calculate the average Stot values for each wavelength
        avg_stot_values = coretype_df[stot_columns].mean().values

        # Plot the average Stot values for the current coretype
        ax.plot(wavelengths, avg_stot_values, marker='o', linestyle='-')
        ax.set_title(f'Spectral Energy Distribution (SED)\n of {coretype_names[coretype]} Coretype', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Wavelength (µm)')
        ax.set_ylabel('Average Stot')

        # Set the x-axis scale to log, display only specified wavelength ticks, and avoid scientific notation
        ax.set_xscale('log')
        ax.set_xticks(wavelengths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

        # Set the y-axis scale to linear, standardize the y-axis range, and avoid scientific notation
        ax.set_ylim(0, y_max)
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Save the figure as a high-quality PDF
    plt.tight_layout()
    plt.savefig(f'Sed Analysis/SED_Analysis_All_Coretypes.pdf', format='pdf', dpi=300)
    plt.show()
    plt.close(fig)

def SED_analysis_single(df, y_max=26):
    wavelengths = [70, 160, 250, 350, 500]
    stot_columns = ['Stot70', 'Stot160', 'Stot250', 'Stot350', 'Stot500']
    coretypes = [1, 2, 3]
    coretype_names = {1: 'Starless', 2: 'Prestellar', 3: 'Protostellar'}

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Loop through each coretype
    for coretype in coretypes:
        # Filter the dataframe for the current coretype
        coretype_df = df[df['Coretype'] == coretype]

        # Calculate the average Stot values for each wavelength
        avg_stot_values = coretype_df[stot_columns].mean().values

        # Plot the average Stot values for the current coretype
        ax.plot(wavelengths, avg_stot_values, marker='o', linestyle='-', label=coretype_names[coretype])

    ax.set_title(f'Spectral Energy Distribution (SED) \nof YSO Coretypes', fontsize=16, fontweight='bold')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Average Stot')

    # Set the x-axis scale to log, display only specified wavelength ticks, and avoid scientific notation
    ax.set_xscale('log')
    ax.set_xticks(wavelengths)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    # Set the y-axis scale to linear, standardize the y-axis range, and avoid scientific notation
    ax.set_ylim(0, y_max)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Add a legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Save the figure as a high-quality PDF
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(f'Sed Analysis/SED_Analysis_Single_Coretypes.pdf', format='pdf', dpi=300)
    plt.show()
    plt.close(fig)

def calculate_discrete_skewness(region_df):
    unique_categories, counts = np.unique(region_df['Coretype'], return_counts=True)
    probabilities = counts / counts.sum()
    mean = np.sum(unique_categories * probabilities)
    std_dev = np.sqrt(np.sum((unique_categories - mean)**2 * probabilities))
    skewness = np.sum((unique_categories - mean)**3 * probabilities) / std_dev**3
    return skewness

def calculate_skewness_and_plot(df):
    # Calculate skewness for each region
    regions = df['Region'].unique()
    skewness_results = {}

    for region in regions:
        region_df = df[df['Region'] == region]
        skewness_results[region] = calculate_discrete_skewness(region_df)

    # Save skewness results to a text file
    with open('Txt_files/skewness_results.txt', 'w') as f:
        for region, skewness in skewness_results.items():
            f.write(f"{region}: {skewness:.3f}\n")

    # Create the barplot
    barWidth = 0.85
    bars = list(skewness_results.values())
    r = np.arange(len(bars))
    colors = ['#FFA07A', '#87CEFA', '#FFDAB9', '#9370DB', '#98FB98', '#FFC0CB', '#B0C4DE']
    
    plt.figure(figsize=(10, 5))
    for i, (region, color) in enumerate(zip(regions, colors)):
        plt.bar(r[i], bars[i], color=color, width=barWidth, edgecolor='white', label=region)

    plt.xticks(r, regions)
    plt.ylabel('Skewness')
    plt.title('Skewness of Distribution of Core Type by Region')
    plt.legend()
    # Add the values on top of the bars
    for i, (region, skewness) in enumerate(skewness_results.items()):
        plt.text(r[i], skewness, f'{skewness:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Save the figure with tight fit and extremely good quality
    plt.savefig('Graphs/skewness_results.pdf', bbox_inches='tight', dpi=300)

    plt.show()




if __name__ == '__main__':
    #variable_skew(df)
    #sig_to_noise()
    #sig_to_poiss()
    #sp_sbg()
    heatmap()
    #boxplot()
    #histogram()
    #chart(res_df_unbalanced, "Unbalanced")
    #chart(res_df_balanced, "Balanced")
    #chart(res_df_Cepheus, "Cepheus")
    #chart(res_df_Ophiuchus, "Ophiuchus")
    #chart(res_df_Taurus, "Taurus")
    #chart(res_df_Corona_Australis, "Corona Australis")
    #chart(res_df_Lupus, "Lupus")
    #chart(res_df_Aquilia, "Aquilia")
    #chart(res_df_Orion, "Orion")

    
    #bar_total_count()
    #labels_all()
    #labels_all_perc()
    #bar_labels()
    #graph_bars()
    #pairplot()
    #heatmap()
    #labels_grouped_bar(df_with_region)
    #labels_grouped_bar_proportion(df_with_region)
    #scatter('Stot250', 'Sp250')
    #cor()
    #print(df.corr().unstack().sort_values(ascending=False).drop_duplicates())
    #plot_nans()
    #plot_balanced_data_region("Balanced")
    #plot_balanced_data_region("Unbalanced")
    #SED_Analysis(df)
    #SED_analysis_combined(df)
    #SED_analysis_single(df)
    #calculate_skewness_and_plot(df_with_region)

    
    """plot_metric("Accuracy", "Accuracy by Balanced Model by Region (K=10)", "Balanced")
    plot_metric("Precision", "Precision by Balanced Model by Region (K=10)", "Balanced")
    plot_metric("Recall", "Recall by Balanced Model by Region (K=10)", "Balanced")
    plot_metric("F1", "F1 by Balanced Model by Region (K=10)", "Balanced")
    
    plot_metric("Accuracy", "Accuracy by Unbalanced Model by Region (K=10)", "Unbalanced")
    plot_metric("Precision", "Precision by Unbalanced Model by Region (K=10)", "Unbalanced")
    plot_metric("Recall", "Recall by Unbalanced Model by Region (K=10)", "Unbalanced")
    plot_metric("F1", "F1 by Unbalanced Model by Region (K=10)", "Unbalanced")"""
