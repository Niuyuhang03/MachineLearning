import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def load_data():
    '''
    读入训练集
    :return:训练集特征
    '''
    train_data = pd.read_csv('train.csv')
    X = train_data.iloc[:, 1:4]
    # 将male和female转为1和0
    labelencoder_X = LabelEncoder()
    X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])
    X = X.values
    # 将缺失数据替换为平均值
    if np.isnan(X.astype(float)).sum() > 0:
        print("NaN exists in train_X.")
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(X)
        X = imp.transform(X).astype(np.int32)
    return X

def normalization(train_X):
    '''
    归一化
    :param train_X: 训练集特征
    :return: 归一化后的训练集特征
    '''
    sc_X = StandardScaler()
    normal_train_X = sc_X.fit_transform(train_X)
    return normal_train_X

def get_init_center(normal_train_X, k):
    '''
    得到k个初始聚类中心
    :param normal_train_X: 归一化后的训练集样本
    :param k: 参数k
    :return: 包含k个聚类中心的数组
    '''
    cluster_center = []
    m_normal_train_X = normal_train_X.shape[0]
    for i in range(k):
        cluster_center.append(normal_train_X[int(random.uniform(0, m_normal_train_X)), :].tolist())
    return cluster_center

def classify(normal_train_X, k):
    '''
    预测
    :param normal_train_X: 归一化后的训练集特征
    :param normal_test_X: 归一化后的测试集特征
    :return:聚类结果
    '''
    cluster_center = get_init_center(normal_train_X, k)
    m_normal_train_X = normal_train_X.shape[0]
    all_index = {}
    changed = True
    cnt = 0
    while changed:
        changed = False
        for i in range(m_normal_train_X):
            cur_index = all_index.get(i, -1)
            min_distance = -1
            min_index = -1
            for j in range(k):
                sq_diff = (normal_train_X[i, :] - np.array(cluster_center[j])) ** 2
                diff = sum(sq_diff.tolist()) ** 0.5
                if min_distance == -1 or diff < min_distance:
                    min_distance = diff
                    min_index = j
            if min_index != cur_index:
                changed = True
                all_index[i] = min_index
        if changed is True:
            for i in range(k):
                sum_dis = []
                for j in range(m_normal_train_X):
                    if all_index[j] == i:
                        sum_dis.append(normal_train_X[i].tolist())
                cluster_center[i] = np.mean(sum_dis, axis=0)
    predict_test_Y = []
    for i in range(m_normal_train_X):
        predict_test_Y.append(all_index[i])
    return np.array(predict_test_Y)

if __name__ == '__main__':
    k = 3
    train_X = load_data()
    normal_train_X = normalization(train_X)

    # 手动实现
    predict_test_Y = classify(normal_train_X, k)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    k_means = KMeans(n_clusters=3, random_state=0)
    k_means.fit(normal_train_X)
    predict_test_Y = k_means.predict(normal_train_X)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)