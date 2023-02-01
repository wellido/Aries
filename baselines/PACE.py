# import time
import os
# import sys
# import datetime
import numpy as np
# import keras
# from joblib import Memory
from tensorflow.keras.models import Model
# import random
# from tensorflow.keras.datasets import mnist
# from numpy import arange
import hdbscan
# import openpyxl
# import argparse
# import tensorflow as tf
# import imageio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from baselines.mmd_critic.run_digits_new import run
# import matplotlib.pyplot as plt
# import cv2
import math


def get_score(x_test, y_test, model):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('预测错的数目：', len(x_test)*(1-score[1]))
    return score


def get_ds(countlist, res_get_s, sample_size, X_test, res, selected_index):
    len_nonnoise = len(X_test) - countlist[0]
    for key in res:
        b = []
        if len(res[key]) > (len_nonnoise/sample_size):
            for num in range(int(round(len(res[key]) / (len_nonnoise/sample_size)))):
                b.append(res[key][res_get_s[key][num]])
        else:
            b.append(res[key][res_get_s[key][0]])

        for i in range(len(b)):
            # X_test2.append(X_test[b[i]])
            # Y_test2.append(Y_test[b[i]])
            selected_index.append(b[i])


def get_std1(X_test, a_unoise, countlist, res, label_noise, first_noise, res_get_s, dis, select_size):
    selected_index = []
    for j in select_size:

        len_noise = j*(1-a_unoise)
        #adaptive random
        selected_index.append(label_noise[first_noise])
        pre_num = []
        pre_num.append(first_noise)
        pre_num.append(np.argmax(dis[first_noise]))
        while len(selected_index) < len_noise:
            mins = []
            for i in range(len(label_noise)):
                if i not in set(pre_num):
                    min_info = [float('inf'), 0, 0]
                    for l in pre_num:
                        if dis[i][l] < min_info[0]:
                            min_info[0] = dis[i][l]
                            min_info[1] = i
                            min_info[2] = l
                    mins.append(min_info)
            maxnum = 0

            selected_index.append(label_noise[mins[0][1]])
            pre_num.append(mins[0][1])
            for i in mins:
                if i[0] > maxnum:

                    selected_index[-1] = label_noise[i[1]]
                    pre_num[-1] = i[1]
                    # pre_num.append(i[1])

        get_ds(countlist, res_get_s, j*a_unoise, X_test, res, selected_index)

    return selected_index


def PACE_selection(model, candidate_data, select_size):
    basedir = os.path.abspath(os.path.dirname(__file__))
    select_layer_idx = -3
    dec_dim = 8
    min_cluster_size = 80
    min_samples = 4
    dense_layer_model = Model(inputs=model.input, outputs=model.layers[select_layer_idx].output)
    dense_output = dense_layer_model.predict(candidate_data)
    minMax = MinMaxScaler()
    dense_output = minMax.fit_transform(dense_output)
    fica = FastICA(n_components=dec_dim)
    dense_output = fica.fit_transform(dense_output)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    r = clusterer.fit(dense_output)
    labels = r.labels_
    y_pred_list = labels.tolist()
    countlist = []

    for i in range(np.min(labels), np.max(labels) + 1):
        countlist.append(y_pred_list.count(i))

    label_noise = []
    for i, l in enumerate(labels):
        if l == -1:
            label_noise.append(i)

    res = {}
    for i, l in enumerate(labels):
        if l != -1:
            if l not in res:
                res[l] = []
            res[l].append(i)

    dis = np.zeros((len(label_noise), len(label_noise)))
    for i in range(len(label_noise)):
        for j in range(len(label_noise)):
            if j != i:
                dis[i][j] = math.sqrt(np.power(dense_output[label_noise[i]] - dense_output[label_noise[j]], 2).sum())

    noise_score = []
    for i, l in enumerate(r.outlier_scores_):
        if labels[i] == -1:
            noise_score.append(l)
    noise_score = np.array(noise_score)
    first_noise = np.argsort(-noise_score)[0]
    res_get_s = {}
    for key in res:
        temp_dense = []
        for l in res[key]:
            temp_dense.append(dense_output[l])
        temp_dense = np.array(temp_dense)
        temp_label = np.full((len(temp_dense)), key)
        mmd_res, _ = run(temp_dense, temp_label, gamma=0.026, m=min(len(temp_dense), 1000), k=0, ktype=0, outfig=None,
                         critoutfig=None, testfile=os.path.join(basedir, 'mmd_critic/data/a.txt'))
        res_get_s[key] = mmd_res
    select_size = [select_size]
    a_unoise = 0.6
    selected_index = get_std1(X_test=candidate_data, a_unoise=a_unoise, countlist=countlist, res=res,
                              label_noise=label_noise, first_noise=first_noise, res_get_s=res_get_s, dis=dis,
                              select_size=select_size)
    return selected_index

