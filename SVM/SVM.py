import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


def load_data():
    '''
    读入训练集和测试集
    :return:训练集特征、交叉验证集特征、训练集标签、交叉验证集标签、测试集特征
    '''
    train_data = pd.read_csv('train.csv')
    X = train_data.iloc[:, 1:4]
    Y = train_data.iloc[:, -1].values
    # 将male，female转为数字
    labelencoder_X = LabelEncoder()
    X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])
    X = X.values
    # 将缺失数据替换为平均值
    if np.isnan(X.astype(float)).sum() > 0:
        print("NaN exists in train_X.")
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(X)
        X = imp.transform(X).astype(np.int32)
    # 划分训练集和交叉验证集，比例为8:2
    train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)

    test_X = pd.read_csv('test.csv').iloc[:, 1:4]
    # 将male，female转为数字
    test_X.iloc[:, 0] = labelencoder_X.transform(test_X.iloc[:, 0])
    test_X = test_X.values
    return train_X, cv_X, train_Y, cv_Y, test_X

def normalization(train_X, cv_X, test_X):
    '''
    归一化
    :param train_X: 训练集特征
    :param cv_X: 交叉验证集特征
    :param test_X：测试集特征
    :return: 归一化后的训练集特征、交叉验证集特征和测试集特征
    '''
    sc_X = MinMaxScaler()
    normal_train_X = sc_X.fit_transform(train_X)
    normal_cv_X = sc_X.transform(cv_X)
    normal_test_X = sc_X.transform(test_X)
    return normal_train_X, normal_cv_X, normal_test_X

def selectJrand(i, m):
    '''
    随机选择alpha
    :param i:alpha
    :param m:alpha参数个数
    :return:
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj,H,L):
    '''
    修剪alpha
    :param aj:alpha值
    :param H:alpha上限
    :param L:alpha下限
    :return:
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(normal_train_X, train_Y, C, toler, maxIter):
    '''
    简化版SMO算法
    :param normal_train_X:归一化后的训练集样本
    :param train_Y:训练集标签
    :param C:惩罚系数
    :param toler:松弛变量
    :param maxIter:最大迭代次数
    :return:
    '''
    normal_train_X = np.mat(normal_train_X)
    train_Y = np.mat(train_Y).transpose()
    b = 0
    m_normal_train_X, n_normal_train_X = np.shape(normal_train_X)
    alphas = np.mat(np.zeros((m_normal_train_X, 1)))
    iter_num = 0
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m_normal_train_X):
            fXi = float(np.multiply(alphas, train_Y).T * (normal_train_X * normal_train_X[i, :].T)) + b
            Ei = fXi - float(train_Y[i])
            if ((train_Y[i] * Ei < -toler) and (alphas[i] < C)) or ((train_Y[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m_normal_train_X)
                # 计算误差Ej
                fXj = float(np.multiply(alphas, train_Y).T * (normal_train_X * normal_train_X[j, :].T)) + b
                Ej = fXj - float(train_Y[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算上下界L和H
                if (train_Y[i] != train_Y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                # 计算eta
                eta = 2.0 * normal_train_X[i, :] * normal_train_X[j, :].T - normal_train_X[i, :] * normal_train_X[i, :].T - normal_train_X[j,:] * normal_train_X[j, :].T
                if eta >= 0:
                    continue
                # 更新alpha_j
                alphas[j] -= train_Y[j] * (Ei - Ej) / eta
                # 修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    continue
                # 更新alpha_i
                alphas[i] += train_Y[j] * train_Y[i] * (alphaJold - alphas[j])
                # 更新b_1和b_2
                b1 = b - Ei - train_Y[i] * (alphas[i] - alphaIold) * normal_train_X[i, :] * normal_train_X[i, :].T - train_Y[j] * (alphas[j] - alphaJold) * normal_train_X[i, :] * normal_train_X[j, :].T
                b2 = b - Ej - train_Y[i] * (alphas[i] - alphaIold) * normal_train_X[i, :] * normal_train_X[j, :].T - train_Y[j] * (alphas[j] - alphaJold) * normal_train_X[j, :] * normal_train_X[j, :].T
                # 根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
    return b, alphas

def get_w(normal_train_X, train_Y, alphas):
    '''
    计算w
    :param normal_train_X:
    :param train_Y:
    :param alphas:
    :return:
    '''
    alphas, dataMat, labelMat = np.array(alphas), np.array(normal_train_X), np.array(train_Y)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, np.shape(dataMat)[1])) * dataMat).T, alphas)
    return w.tolist()

def classify(normal_train_X, train_Y, normal_test_X):
    b, alphas = smoSimple(normal_train_X, train_Y, 0.6, 0.001, 4000)
    w = get_w(normal_train_X, train_Y, alphas)
    predict_test_Y = np.dot(np.mat(normal_test_X), np.mat(w)) + b
    return predict_test_Y.flatten().tolist()[0]

if __name__ == '__main__':
    train_X, cv_X, train_Y, cv_Y, test_X = load_data()
    normal_train_X, normal_cv_X, normal_test_X = normalization(train_X, cv_X, test_X)

    # 手动实现
    predict_test_Y = classify(normal_train_X, train_Y, normal_test_X)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    print("--------------")
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(normal_train_X, train_Y)
    score = classifier.score(normal_cv_X, cv_Y)
    predict_cv_Y = classifier.predict(normal_cv_X)
    print("score=" + str(score))
    print("--------------")
    print("Predict Test Dataset:")
    predict_test_Y = classifier.predict(normal_test_X)
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)