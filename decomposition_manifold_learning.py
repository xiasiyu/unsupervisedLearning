import numpy as np
import pandas as pd
import pickle, gzip

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE

from scatter import scatterPlot

color = sns.color_palette()


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')

f.close()

X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

train_index = range(0, len(X_train))
validation_index = range(len(X_train), len(X_train) + len(X_validation))
test_index = range(len(X_train) + len(X_validation), len(X_train) + len(X_validation) + len(X_test))

X_train = pd.DataFrame(data=X_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

X_validation = pd.DataFrame(data=X_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

X_test = pd.DataFrame(data=X_test, index=test_index)
y_test = pd.Series(data=y_train, index=test_index)


#isoMap  等距映射,测地距离而不是欧几里距离
n_neighbors = 5
n_components = 10
n_jobs = 4

isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs)
isomap.fit(X_train.loc[0:5000, :])
X_train_isomap = isomap.transform(X_train)
X_train_isomap = pd.DataFrame(data=X_train_isomap, index=train_index)
scatterPlot(X_train_isomap, y_train, "Isomap")


#MDS 多维标度法
n_components = 2
n_init = 12
max_iter = 1200
metric = True
n_jobs = 4
random_state = 2018

mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, metric=metric, n_jobs=n_jobs,
          random_state=random_state)
X_train_mds = mds.fit_transform(X_train.loc[0:1000, :])
X_train_mds = pd.DataFrame(data=X_train_mds, index=train_index[0:1001])
scatterPlot(X_train_mds, y_train, "Multidimensional Scaling")



#LLE 局部线性嵌入
n_neighbors = 10
n_components = 2
method = "modified"
n_jobs = 4
random_state = 2018

lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method,
                                   random_state=random_state, n_jobs=n_jobs)

lle.fit(X_train.loc[0:5000, :])
X_train_lle = lle.transform(X_train)
X_train_lle = pd.DataFrame(data=X_train_lle, index=train_index)
scatterPlot(X_train_lle, y_train, "Locally Linear Embedding")



#t-SNE t-分布随机领域嵌入
n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = "random"
random_state = 2018

tsne = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity,
            early_exaggeration=early_exaggeration, init=init, random_state=random_state)


#PCA before t-SNE
n_components = 784
whiten = False
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

X_train_tsne = tsne.fit_transform(X_train_PCA.loc[:5000, :9])
X_train_tsne = pd.DataFrame(data=X_train_tsne, index=train_index[:5001])
scatterPlot(X_train_tsne, y_train, "t-SNE")




