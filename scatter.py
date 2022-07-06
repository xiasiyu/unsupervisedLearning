import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
color = sns.color_palette()

def scatterPlot(xDF, yDF, algoName):
    plt.clf()
    temp = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    temp = pd.concat((temp, yDF), axis=1, join="inner")
    temp.columns = ['First Vector', 'Second Vector', 'Label']
    sns.lmplot(x="First Vector", y="Second Vector", hue='Label', data=temp, fit_reg=False)
    ax = plt.gca()
    ax.set_title("sepration of observation using " + algoName)
    plt.savefig('./out/%s.jpg' % algoName)
