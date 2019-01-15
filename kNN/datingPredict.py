# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import operator


def reading_train_file():
    """读入训练集
    :return: 特征集
             标签集
    """
    fr = open("datingTestSet.txt")
    contents = fr.readlines()
    number_of_lines = len(contents)
    return_mat = np.zeros((number_of_lines, 3))
    labels = []
    index = 0
    for line in contents:
        line = line.strip()  # 删除"\n" "\t" "\r" " "等
        list_of_line = line.split('\t')
        return_mat[index, :] = list_of_line[0:3]
        if list_of_line[-1] == "didntLike":
            labels.append(-1)
        elif list_of_line[-1] == "smallDoses":
            labels.append(0)
        elif list_of_line[-1] == "largeDoses":
            labels.append(1)
        index += 1
    return return_mat, labels


def show_data(dating_data_mat, dating_labels):
    """数据可视化

    :param dating_data_mat: 特征集
    :param dating_labels: 标签集
    :return: None
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))  # 把画布fig分成2行2列，总大小为13*8，不共享x轴和y轴
    number_of_lines = len(dating_labels)
    labels_colors = []
    for label in dating_labels:
        if label == -1:
            labels_colors.append('black')
        elif label == 0:
            labels_colors.append('orange')
        elif label == 1:
            labels_colors.append('red')

    axs[0][0].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 1], color=labels_colors, s=15, alpha=0.5) # 画特征1和特征2的图，散点大小15，透明度0.5
    axs[0][1].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=0.5)
    axs[1][0].scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=0.5)

    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')  # 设置图例
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()


def normal_mat(dating_data_array, test_list):
    """归一化
    :param dating_data_array: 特征集
    :param test_list: 测试队列
    :return: 归一化的特征集和测试队列
    """
    number_of_lines = dating_data_array.shape[0]
    test_array = np.array(test_list)
    array = np.vstack((dating_data_array, test_array))
    max_list = array.max(0)
    min_list = array.min(0)
    array = (array - np.tile(min_list, (number_of_lines + 1, 1))) / (np.tile(max_list - min_list, (number_of_lines + 1, 1)))
    return array[:-1, :], array[-1, :].tolist()


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


def classify_test(dating_data_array, dating_labels, k):
    rate = 0.1
    lines_of_dating_data = dating_data_array.shape[0]
    cross_validation_number = int(rate * lines_of_dating_data)
    error_num = 0
    for i in range(cross_validation_number):
        label = classify(dating_data_array[i, :], dating_data_array[cross_validation_number:lines_of_dating_data, :], dating_labels[cross_validation_number:lines_of_dating_data], k)
        if label != dating_labels[i]:
            error_num += 1
    print("error rate:" + str(int(error_num / cross_validation_number * 1000) / 10) + "%")


if __name__ == "__main__":
    like_degree = ["smallDoses", "largeDoses", "didntLike"]
    k = 3
    test_list = [44000, 12, 0.5]

    dating_data_array, dating_labels = reading_train_file()
    # show_data(dating_data_array, dating_labels)
    normal_dating_data_array, normal_test_list = normal_mat(dating_data_array, test_list)
    classify_test(normal_dating_data_array, dating_labels, k)
    label = classify(normal_test_list, normal_dating_data_array, dating_labels, k)
    print(like_degree[label])