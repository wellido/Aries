import numpy as np
import tensorflow as tf
from scipy import stats
from tensorflow_addons.optimizers import SGDW
import tensorflow_model_optimization as tfmot
tf.keras.optimizers.SGDW = SGDW


def Aries_estimation(candidate_data, drop_model=None, base_map=None, base_acc=None, section_num=50):
    mode_his = []
    consistant_list = []
    BALD_list = []
    mode_list = []
    for _ in range(section_num):
        prediction = np.argmax(drop_model.predict(candidate_data), axis=1)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(len(candidate_data)):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1, ))[1][0]
        mode_list.append(mode_num)
    mode_list = np.asarray(mode_list)
    for i in range(1, section_num + 1):
        consistant_idxs = np.where(mode_list == i)[0]
        if len(consistant_idxs) > 0:
            mode_his.append(i)
            consistant_list.append(len(consistant_idxs))
    mode_his = np.asarray(mode_his)
    consistant_list = np.asarray(consistant_list)
    base_mode = base_map[0]
    base_consistent = base_map[1]
    base_probability = base_map[2]
    base_mode = np.asarray(base_mode)
    base_consistent = np.asarray(base_consistent)
    base_probability = np.asarray(base_probability)
    xy, x_ind, y_ind = np.intersect1d(base_mode, mode_his, return_indices=True)
    consistant_list = consistant_list[y_ind]
    base_consistent = base_consistent[x_ind]
    base_probability = base_probability[x_ind]
    acc1 = np.sum(consistant_list * base_probability)
    acc2 = base_acc * (consistant_list[-1] / base_consistent[-1]) * len(candidate_data)
    estimated_acc = (acc1 + acc2) / 2
    print(estimated_acc)
    return estimated_acc, acc1, acc2


def drop_check_func(ori_model, drop_model, x_test, y_test, section_num):
    intersection_list = []
    consistant_list = []
    right_list = []
    mode_his = []
    y_test = y_test.reshape(1, -1)[0]
    ori_predictions = ori_model.predict(x_test)
    ori_predict_label = np.argmax(ori_predictions, axis=1)
    right_label = np.where(ori_predict_label == y_test)[0]
    # print(len(right_label))
    BALD_list = []
    mode_list = []
    for _ in range(section_num):
        prediction = np.argmax(drop_model.predict(x_test), axis=1)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(len(x_test)):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1, ))[1][0]
        mode_list.append(mode_num)
    mode_list = np.asarray(mode_list)
    for i in range(1, section_num + 1):
        consistant_idxs = np.where(mode_list == i)[0]
        if len(consistant_idxs) > 0:
            mode_his.append(i)
            intersection_idx = np.intersect1d(consistant_idxs, right_label)
            consistant_list.append(len(consistant_idxs))
            right_list.append(len(right_label))
            intersection_list.append(len(intersection_idx))
    print(mode_his)
    print(consistant_list)
    # print(right_list)
    print(intersection_list)
    mode_his = np.asarray(mode_his)
    consistant_list = np.asarray(consistant_list)
    intersection_list = np.asarray(intersection_list)
    pro_list = intersection_list / consistant_list
    return [mode_his, consistant_list, pro_list]






