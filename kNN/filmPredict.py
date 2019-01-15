# -*- coding: UTF-8 -*-
import numpy as np
import operator


def create_data_set():
    """创建数据集
    创建数据集作为训练集

    Parameters:
        None
    Returns:
        group - 数据集
        labels - 分类标签
    """
    group = np.array([[98, 3], [81, 2], [6, 78], [1, 103]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def classify(inX, data_set, labels, k):
    """k-邻近算法
    统计距离测试点最近的k个样本的labels，取最多出现的label作为预测label

    Parameters:
        inX - 测试数据，1*n维
        data_set - 训练集
        labels - 训练集分类标签
        k - KNN参数，取最近的k个样本
    Returns:
        sorted_classify_count[0][0] - 出现次数最多的label
    """
    data_set_size = data_set.shape[0]  # 取出样本数
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set  # 将测试数据横向复制一遍（1*n），纵向复制data_set_size遍（data_set_size*n）后相减
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)  # 各行相加，axis=1按行相加，axis=0按列相加
    distances = sq_distances ** 0.5
    sorted_distances_index = distances.argsort()  # 将距离向量从小到大排序的索引值求出
    classify_count = {}  # 记录样本出现次数的字典
    for i in range(k):
        label = labels[sorted_distances_index[i]]
        classify_count[label] = classify_count.get(label, 0) + 1  # 如果该label不存在，返回默认值0
    sorted_classify_count = sorted(classify_count.items(), key=operator.itemgetter(1), reverse=True)  # 排序字典，itemgetter(1)以值排序，itemgetter(0)以键排序，倒序
    return sorted_classify_count[0][0]  # {(labels, count), (labels, count), ...}

if __name__ == '__main__':
    group, labels = create_data_set()
    test = [2, 78]
    result_label = classify(test, group, labels, 2)
    print(result_label)