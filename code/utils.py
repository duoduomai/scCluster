import scanpy as sc
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn import metrics

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc = np.round(cluster_acc(y_true, y_pred),5)
    ami = np.round(metrics.adjusted_mutual_info_score(y_true, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    return acc, ami, nmi, ari

def get_clusters(X, res, n):
    adata_latent = sc.AnnData(X)
    if adata_latent.shape[0] > 200000:
        np.random.seed(adata_latent.shape[0])
        adata_latent = adata_latent[np.random.choice(adata_latent.shape[0], 200000, replace=False)]
    sc.pp.neighbors(adata_latent, n_neighbors=n, use_rep="X")
    sc.tl.louvain(adata_latent, resolution=res)
    y_pred_init = np.asarray(adata_latent.obs['louvain'], dtype=int)
    n_clusters = np.unique(y_pred_init).shape[0]
    if n_clusters <= 1:
        exit("Error: Only one cluster detected. Please choose a larger resolution.")
    else:
        print("Estimated number of clusters:", n_clusters)
    return n_clusters



