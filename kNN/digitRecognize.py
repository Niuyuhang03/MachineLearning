# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
    """读单个文件到一维矩阵
    :param filename: 文件名
    :return: 特征矩阵
    """
    return_mat = np.zeros((1, 1024))
    fr = open(filename)
    content = fr.readlines()
    content_lines = len(content)
    for i in range(content_lines):
        for j in range(32):
            return_mat[0, i*32+j] = content[i][j]
    return return_mat


def handwritingClassTrain():
    """读训练文件
        :return: 特征集、标签集
    """
    labels = []
    train_file_list = listdir("trainingDigits")
    train_file_num = len(train_file_list)
    digit_data_mat = np.zeros((train_file_num, 1024))
    for i in range(train_file_num):
        filename = train_file_list[i]
        labels.append(filename.split('_')[0])
        digit_data_mat[i, :] = img2vector("trainingDigits\%s" % filename)
    return digit_data_mat, labels


def handwritingClassTest():
    """读测试文件
    :return: 特征集、标签集
    """
    labels = []
    train_file_list = listdir("testDigits")
    train_file_num = len(train_file_list)
    digit_data_mat = np.zeros((train_file_num, 1024))
    for i in range(train_file_num):
        filename = train_file_list[i]
        labels.append(filename.split('_')[0])
        digit_data_mat[i, :] = img2vector("testDigits\%s" % filename)
    return digit_data_mat, labels


def classify(inX, data_set, labels, k):
    """KNN分类
    :param inX:预测矩阵，1*n
    :param data_set: 特征矩阵
    :param labels: 标签集
    :param k: KNN参数k，最近点数目
    :return: 预测label
    """
    line_of_data_set = data_set.shape[0]
    sq_diff = (np.tile(inX, (line_of_data_set, 1)) - data_set) ** 2
    diff = (sq_diff.sum(axis=1)) ** 0.5
    sorted_diff_index = diff.argsort()
    predict_label = {}
    for i in range(k):
        label = labels[sorted_diff_index[i]]
        predict_label[label] = predict_label.get(label, 0) + 1
    sorted_predict_label = sorted(predict_label.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_predict_label[0][0]


def classifyTest(inX, inlabels, data_set, labels, k):
    """分类验证器
    :param inX: 测试集特征
    :param inlabels: 测试集标签
    :param data_set: 训练集特征
    :param labels: 训练集标签
    :param k: KNN参数k，取最近k个样本
    :return: None
    """
    n_samples = len(inX)
    error_num = 0
    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(data_set, labels)
    for i in range(n_samples):
        label = neigh.predict(inX[i, :].reshape(1, 1024))
        # label = classify(inX[i, :], data_set, labels, k)
        if inlabels[i] != label:
            error_num += 1
    print("error rate:%f%%" % (error_num/n_samples*100))


if __name__ == "__main__":
    k = 3
    digit_data_mat, labels = handwritingClassTrain()
    test_data_mat, test_labels = handwritingClassTest()
    classifyTest(test_data_mat, test_labels, digit_data_mat, labels, k)