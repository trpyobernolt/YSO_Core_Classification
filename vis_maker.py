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

plt.rcParams["font.family"] = "serif"

"""plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})"""

df = pd.read_csv("Data/clean_data.csv")


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
    figure.savefig("Graphs/heatmap.png")

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
    labels = {'Starless':(df['Coretype'] == 1).sum(),
    'Prestellar': (df['Coretype'] == 2).sum(),
    'Protostellar': (df['Coretype'] == 3).sum()}
    plt.bar(labels.keys(), labels.values())
    bar_width = .1
    for i, key in enumerate(labels.keys()):
        x = 10*i * bar_width + bar_width / 2
        plt.text(x, labels[key] + 40, '{}'.format(labels[key]), ha='center')
    plt.title("Count of Coretypes")
    save_path = 'Graphs/Bars/Coretype.pdf'
    plt.savefig(save_path)
    plt.show()
    plt.clf()
    
results_list = []
"""Results take the form of: [Name, Precision, Recall, F1, Accuracy]"""
results_list.append(["NB", 0.73, 0.74, 0.74, 0.743639921722113])
results_list.append(["Perceptron", 0.87, 0.87, 0.87, 0.868884540117416])
results_list.append(["SVM", 0.89, 0.89, 0.89, 0.886497064579256])
results_list.append(["SGD", 0.88, 0.88, 0.88, 0.879321591650358])
results_list.append(["Passive Aggressive", 0.87, 0.87, 0.87, 0.874103065883887])
results_list.append(["LDA", 0.63, 0.64, 0.59, 0.639921722113503])
results_list.append(["QDA", 0.82, 0.81, 0.8, 0.806914546640574])
results_list.append(["KNN", 0.86, 0.86, 0.86, 0.858447488584474])
results_list.append(["Decision Tree", 0.87, 0.87, 0.87, 0.868232224396608])
res_df = pd.DataFrame(results_list, columns =['Name', 'Precision', 'Recall', 
                                              'F1', 'Accuracy'])  
colors = {'NB': 'blue', 'Perceptron':'darkturquoise', 'SVM':'darkorange', 'SGD':'red',
          'Passive Aggressive':'purple', 'LDA':'brown', 'QDA':'pink',
          'KNN':'gray', 'Decision Tree':'green'}


def chart_precision():
    fig, ax = plt.subplots()
    dummy = res_df.sort_values(by=['Precision'], ascending=False)
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
    plt.ylim(0.55, 0.9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
          bar.get_height(),
          horizontalalignment='center',
          weight='bold'
          )
    plt.tick_params(axis='x', labelbottom = False, bottom=False)
    plt.title("Model Precision")
    
    c = dummy['Name'].apply(lambda x: colors[x])
    ax = plt.subplot(111) #specify a subplot

    bars = ax.bar(dummy['Name'], dummy['Precision'], color=c) #Plot data on subplot axis

    for i, j in colors.items(): #Loop over color dictionary
        ax.bar(dummy['Name'], dummy['Precision'],width=0,color=j,label=i) #Plot invisible bar graph but have the legends specified
    plt.legend(loc=1, prop={'size': 6})
    
    plt.savefig("Graphs/Results/Precision.pdf")
    return fig

def chart_recall():
    fig, ax = plt.subplots()
    dummy = res_df.sort_values(by=['Recall'], ascending=False)
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
    plt.ylim(0.55, 0.9)
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
    
    plt.title("Model Recall")
    plt.savefig("Graphs/Results/Recall.pdf")
    return plt
    
def chart_f1():
    fig, ax = plt.subplots()
    dummy = res_df.sort_values(by=['F1'], ascending=False)
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
    plt.ylim(0.55, 0.9)
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
    
    plt.title("Model F1")
    plt.savefig("Graphs/Results/F1.pdf")
    return plt

def chart_accuracy():
    fig, ax = plt.subplots()
    dummy = res_df.sort_values(by=['Accuracy'], ascending=False)
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
    plt.ylim(0.55, 0.9)
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
    
    plt.title("Model Accuracy")
    plt.savefig("Graphs/Results/Accuracy.pdf")
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
            





if __name__ == '__main__':
    """sig_to_noise()
    sig_to_poiss()
    sp_sbg()
    heatmap()
    boxplot()
    histogram()
    chart_precision()
    chart_recall()
    chart_f1()
    chart_accuracy()"""
    #labels_all()
    #bar_labels()
    #graph_bars()
    pairplot()
    
    
    
    
    
    