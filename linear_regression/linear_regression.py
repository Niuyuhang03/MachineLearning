import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

def load_data():
    '''
    读入训练集和测试集
    :return:训练集特征、交叉验证集特征、训练集标签、交叉验证集标签、测试集特征
    '''
    train_data = pd.read_csv('train.csv')
    X = train_data.iloc[:, :4]
    Y = train_data.iloc[:, -1].values
    # 将State转为数字
    labelencoder_X = LabelEncoder()
    X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])
    X = X.values
    # 将缺失数据替换为平均值
    if np.isnan(X.astype(float)).sum() > 0:
        print("NaN exists in train_X.")
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(X)
        X = imp.transform(X).astype(np.int32)
    # 划分训练集和交叉验证集，比例为8:2
    train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)

    test_X = pd.read_csv('test.csv')
    # 将State转为数字
    test_X.iloc[:, 3] = labelencoder_X.transform(test_X.iloc[:, 3])
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
    mc_X = MinMaxScaler()
    normal_train_X = mc_X.fit_transform(train_X)
    normal_cv_X = mc_X.transform(cv_X)
    normal_test_X = mc_X.transform(test_X)
    return normal_train_X, normal_cv_X, normal_test_X

def gradient_descent(normal_train_X, train_Y, alpha, num_iters):
    '''
    梯度下降求theta
    :param normal_train_X:归一化后的训练集特征
    :param train_Y:训练集标签
    :param alpha:学习速率
    :param num_iters:迭代次数
    :return:回归系数theta
    '''
    num_train_Y = train_Y.shape[0]
    m_normal_train_X = normal_train_X.shape[1]
    theta = np.zeros((m_normal_train_X,1))
    for iter in range(num_iters):
        new_theta = np.zeros((m_normal_train_X,1))
        for j in range(m_normal_train_X):
            new_theta[j] =  np.dot(np.transpose(normal_train_X[:, j]), (np.dot(normal_train_X, theta) - np.transpose([train_Y])))
        theta -= new_theta * alpha / num_train_Y
    return theta

def classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters):
    '''
    预测
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_test_X: 归一化后的测试集特征
    :param alpha：学习速率
    :param num_iters：迭代次数
    :return: 测试集标签，theta
    '''
    theta = gradient_descent(normal_train_X, train_Y, alpha, num_iters)
    predict_test_Y = np.dot(normal_test_X, theta).flatten()
    return  predict_test_Y, theta

def cv_classify(theta, normal_cv_X, cv_Y):
    '''
    预测交叉验证集，选取最优k值
    :param theta：回归系数theta
    :param normal_cv_X: 归一化后的交叉验证集特征
    :param cv_Y：交叉验证集表标签
    '''
    print("--------------")
    print("Test CV:")
    predict_cv_Y = np.dot(normal_cv_X, theta).flatten()
    score = f1_score(cv_Y.astype(np.float64), predict_cv_Y.astype(np.float64))
    cm = confusion_matrix(cv_Y, predict_cv_Y)
    print(cm)
    print("f1_score="+str(score))

if __name__ == '__main__':
    train_X, cv_X, train_Y, cv_Y, test_X = load_data()
    normal_train_X, normal_cv_X, normal_test_X = normalization(train_X, cv_X, test_X)
    alpha = 0.01
    num_iters = 10000

    # 手动实现
    predict_test_Y, theta = classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters)
    # cv_classify(theta, normal_cv_X, cv_Y)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    print("--------------")
    classifier = LinearRegression()
    classifier.fit(normal_train_X, train_Y)
    score = classifier.score(normal_cv_X, cv_Y)
    predict_cv_Y = classifier.predict(normal_cv_X)
    print("f1_score=" + str(score))
    # cm = confusion_matrix(cv_Y, predict_cv_Y)
    # print(cm)
    print("--------------")
    print("Predict Test Dataset:")
    predict_test_Y = classifier.predict(normal_test_X)
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)