from Bio.Cluster import kcluster
from Bio.Cluster import clustercentroids

from sklearn.metrics import silhouette_score
from create_quries import create_queries
from utils import *

import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime

import matplotlib.pyplot as plt


def get_clusters(npass=50):
    """
    运行聚类模块，npass为以随机中心点重复运行的次数
    """
    fut_pred = 7
    start_time = datetime(2021, 1, 1, 0, 0)
    end_time = datetime(2021, 2, 5, 0, 0)
    query_file = create_queries(start_time, end_time)

    with open(query_file, 'r') as f:
        lines = f.readlines()

    col_set = set()
    for i in range(0, len(lines), 3):
        cols = lines[i + 2].split(' ')
        for col in cols:
            if len(col.strip()) > 0:
                col_set.add(col)

    # print(col_set)

    col_arrivals = {}
    col_day_arrival = {}
    for col in col_set:
        col_arrivals[col] = []
        col_day_arrival[col] = 0

    day_index = -1
    pre_q_time = datetime(1, 1, 1, 0, 0, 0)
    for i in range(0, len(lines), 3):
        q_time_str = lines[i]
        q_time = get_datetime(q_time_str)
        if q_time.date() != pre_q_time.date() and pre_q_time.year != 1:
            for col in col_arrivals.keys():
                col_arrivals[col].append(col_day_arrival[col])
                col_day_arrival[col] = 0

        q_query_str = lines[i + 1]

        q_index_col = lines[i + 2].split(' ')
        for col in q_index_col:
            if len(col.strip()) > 0:
                col_day_arrival[col] += 1

        pre_q_time = q_time

    # print(col_arrivals)

    for col in col_arrivals.keys():
        col_arrivals[col].append(col_day_arrival[col])

    q_data = DataFrame(col_arrivals).T
    q_data_kmeans = q_data.to_numpy()[:, :-fut_pred]
    # print(q_data)

    q_time_cnt = q_data_kmeans.shape[1]

    sil_scores = []
    K_LOWER = 3
    K_UPPER = 15
    for param_k in range(K_LOWER, K_UPPER):
        cluster_id, error, n_found = kcluster(q_data_kmeans, param_k, dist='u', npass=npass)
        # sil_avg = silhouette_score(q_data_np, cluster_id)
        sil_avg = silhouette_score(q_data_kmeans, cluster_id, metric='cosine')
        sil_scores.append(sil_avg)

    e = [i + K_LOWER for i, j in enumerate(sil_scores) if j == max(sil_scores)]
    plt.title("Clustering K vs Silhouette Score")
    plt.xlabel("Clustering K")
    plt.ylabel("Silhouette Score")
    plt.plot(range(K_LOWER, K_UPPER), sil_scores)
    plt.show()

    cluster_id, error, n_found = kcluster(q_data_kmeans, e[0], dist='u', npass=npass)
    # print(cluster_id)
    q_cluster = []
    for i in range(e[0]):
        # ids = np.where(cluster_id == i)
        # print(ids)
        # print(q_data.take(ids, axis=0))
        q_cluster.append(DataFrame(q_data.iloc[np.where(cluster_id == i)]))
    # print(q_cluster)
    print(q_cluster)
    return query_file, q_cluster
