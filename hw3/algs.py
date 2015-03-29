import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def km(ds):
    pipeline = Pipeline([('scaling', StandardScaler(with_mean=False)), 
                         ('k-means', KMeans(n_clusters=ds.N))])
    estimator = pipeline.fit(ds.X)
    return estimator


def em(ds):
    gmm = GMM(n_components=ds.N)
    gmm.fit(ds.X)
    return gmm


def pca(ds):
    N = int(np.ceil(len(ds.X[0])/2.))
    pipeline = Pipeline([('scaling', StandardScaler()), 
                         ('pca', PCA(n_components=N))])
    estimator = pipeline.fit(ds.X)
    return estimator


def ica(ds):
    N = int(np.ceil(len(ds.X[0])/2.))
    pipeline = Pipeline([('scaling', StandardScaler(with_mean=False)), 
                         ('ica', FastICA(n_components=N))])
    estimator = pipeline.fit(ds.X)
    return estimator

 
def rca(ds):
    N = int(np.ceil(len(ds.X[0])/2.))
    pipeline = Pipeline([('scaling', StandardScaler()),
                         ('rca', RandomizedPCA(n_components=N))])
    estimator = pipeline.fit(ds.X)
    return estimator


def lda(ds):
    N = int(np.ceil(len(ds.X[0])/2.))
    pipeline = Pipeline([('scaling', StandardScaler()),
                         ('lda', LDA(n_components=N))])
    estimator = pipeline.fit(ds.X, ds.Y)
    return estimator


