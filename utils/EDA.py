import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# styling
# plt.style.use("ggplot")
rcParams['figure.figsize'] = (30, 15)

# plot histogram
def histogram(data):
    data.hist()
    plt.savefig('histogram.png')
    plt.show()

# create 8x3 subplot
def bivariate_analysis(data):
    plt.subplots(8, 3, figsize=(30,30))

    # plot density plot for each variable
    for idx, col in enumerate(data.columns):
        ax = plt.subplot(8,3, idx + 1)
        ax.yaxis.set_ticklabels([])

        sns.distplot(data.loc[data['target'] == 0][col], hist=False, axlabel=False, kde_kws={'linestyle':'-', 'color':'black', 'label':"Not credible"})
        sns.distplot(data.loc[data['target'] == 1][col], hist=False, axlabel=False, kde_kws={'linestyle':'--', 'color':'black', 'label':"Credible"})

        ax.set_title(col)

    # hide last subplot
    plt.subplot(8, 3, 24).set_visible(False)
#     plt.legend(data['target'])
    plt.savefig('subplots.png')
    plt.show()

# plot heatmap 
def get_correlation(data):
    corr_mat = data.corr()
    heat_map = sns.heatmap(corr_mat, 
                     cbar=True, 
                     annot=True, 
                     square=True, 
                     fmt='.2f', 
                     annot_kws={'size': 10}, 
                     yticklabels=data.columns, 
                     xticklabels=data.columns, 
                     cmap="Spectral_r")
    plt.savefig('heatmap.png')
    plt.show()

