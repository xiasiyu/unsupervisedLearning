import numpy as np
import pandas as pd
import pickle, gzip

import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


from scatter import scatterPlot

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')

f.close()

X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

# print("Shape of X_train: ", X_train.shape)
# print("Shape of y_train: ", y_train.shape)
# print("Shape of X_validation: ", X_validation.shape)
# print("Shape of y_validation: ", y_validation.shape)
# print("Shape of X_test: ", X_test.shape)
# print("Shape of y_test: ", y_test.shape)

train_index = range(0, len(X_train))
validation_index = range(len(X_train), len(X_train) + len(X_validation))
test_index = range(len(X_train) + len(X_validation), len(X_train) + len(X_validation) + len(X_test))

X_train = pd.DataFrame(data=X_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

X_validation = pd.DataFrame(data=X_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

X_test = pd.DataFrame(data=X_test, index=test_index)
y_test = pd.Series(data=y_train, index=test_index)


# print(X_train.describe())
# print(y_train.head())


def view_digits(example):
    label = y_train.loc[0]
    image = X_train.loc[example, :].values.reshape([28, 28])
    plt.title('Example: %d Label:%d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


# view_digits(0)

# def scatterPlot(xDF, yDF, algoName):
#     plt.clf()
#     temp = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
#     temp = pd.concat((temp, yDF), axis=1, join="inner")
#     temp.columns = ['First Vector', 'Second Vector', 'Label']
#     sns.lmplot(x="First Vector", y="Second Vector", hue='Label', data=temp, fit_reg=False)
#     ax = plt.gca()
#     ax.set_title("sepration of observation using " + algoName)
#     plt.savefig('./out/%s.jpg' % algoName)

n_components = 784
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

scatterPlot(X_train_PCA, y_train, "PCA")


#kernel PCA
n_components = 100
kernel = "rbf"
gamma = None
random_state = 2018
n_jobs = 1

kernelPCA = KernelPCA(n_components = n_components, kernel = kernel, gamma = gamma,
                      n_jobs=n_jobs, random_state= random_state)

kernelPCA.fit(X_train.loc[:10000, :])

X_train_kernel = kernelPCA.transform(X_train)
X_train_kernel = pd.DataFrame(data=X_train_kernel, index=train_index)

scatterPlot(X_train_kernel, y_train, "KernelPCA")

#SVD
n_components = 200
algorithm = "randomized"
n_iter = 5
random_state=2018

svd = TruncatedSVD(n_components = n_components, algorithm = algorithm,
                   n_iter = n_iter, random_state= random_state)
X_trains_svd = svd.fit_transform(X_train)
X_trains_svd = pd.DataFrame(data=X_trains_svd, index=train_index)

scatterPlot(X_trains_svd, y_train, "Singular Value Decomposition")


#Gauss random projection

n_components = "auto"
eps = 0.5
random_state = 2018

GRP = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)

X_train_grp = GRP.fit_transform(X_train)
X_train_grp = pd.DataFrame(data=X_train_grp, index=train_index)

scatterPlot(X_train_grp, y_train, "Gaussian Random Projection")


#Sparse randome projection
n_components = "auto"
density = "auto"
eps = 0.5
dense_output = False
random_state = 2018

SRP = SparseRandomProjection(n_components=n_components, density=density, eps=eps, dense_output=dense_output,
                                    random_state=random_state)

X_train_srp = SRP.fit_transform(X_train)
X_train_srp = pd.DataFrame(data=X_train_srp, index=train_index)
scatterPlot(X_train_srp, y_train, "Sparse Random Projection")
