from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import torch
def kmeans(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for i in range(repeat):
        seed=i
        np.random.seed(seed)
        torch.manual_seed(seed)
        kmeans = KMeans(n_clusters=n_clusters,random_state=i)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)

        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    s1 = sum(nmi_list) / len(nmi_list)
    s2 = sum(ari_list) / len(ari_list)
    # print('\t[Clustering] NMI: {:.4f}'.format(s1))
    # print('\t[Clustering] ARI: {:.4f}'.format(s2))
    
