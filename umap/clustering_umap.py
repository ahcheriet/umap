import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import h5py
import numpy as np
import metrics

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# Datasets
d = pd.read_csv("Your_path_Datasets/mnist_784.csv")

time_start = time.time()
standard_embedding, y_pred = umap.UMAP(n_neighbors=5,n_clusters=10, metric='correlation').fit_transform(d.iloc[:, :783])
y_pred=np.argmax(y_pred, axis=1)
time_end = time.time()
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=y_pred, s=1, cmap='Spectral');

plt.show();
acc = np.round(cluster_acc(d.iloc[:, 784], y_pred),5)
print("Accuracy: ",acc)
print("NMI: ",normalized_mutual_info_score(d.iloc[:, 784], y_pred))
print('UMAP runtime: {} seconds'.format(time_end-time_start))
