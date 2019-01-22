import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
    mc_X = MinMaxScaler()
    normal_train_X = mc_X.fit_transform(train_X)
    normal_cv_X = mc_X.transform(cv_X)
    normal_test_X = mc_X.transform(test_X)
    return normal_train_X, normal_cv_X, normal_test_X

def gradient_descent(normal_train_X, train_Y, alpha, num_iters, lamb):
    '''
    梯度下降求theta
    :param lamb：正则化系数
    :param normal_train_X:归一化后的训练集特征
    :param train_Y:训练集标签
    :param alpha:学习速率
    :param num_iters:迭代次数
    :return:回归系数theta
    '''
    num_train_Y = train_Y.shape[0]
    m_normal_train_X = normal_train_X.shape[1]
    add_mar = np.tile([1],(num_train_Y, 1))
    normal_train_X_add = np.hstack((add_mar, normal_train_X))
    theta = np.zeros((m_normal_train_X + 1,1))
    for iter in range(num_iters):
        error = np.zeros((m_normal_train_X + 1,1))
        for j in range(m_normal_train_X + 1):
            error[j] =  np.dot(np.transpose(normal_train_X_add[:, j]), (sigmoid(np.dot(normal_train_X_add, theta)) - np.transpose([train_Y]))) + lamb / num_train_Y * theta[j]
        error[0] -= lamb / num_train_Y * theta[0]
        theta -= error * alpha / num_train_Y
    return theta

def sigmoid(z):
    '''
    sigmoid函数
    :param z:
    :return: sigmoid(z)
    '''
    return 1.0 / (1 + np.exp(-z))

def classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters, lamb):
    '''
    预测
    :param lamb：正则化系数
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_test_X: 归一化后的测试集特征
    :param alpha：学习速率
    :param num_iters：迭代次数
    :return: 测试集标签，theta
    '''
    theta = gradient_descent(normal_train_X, train_Y, alpha, num_iters, lamb)
    add_mar = np.tile([1], (normal_test_X.shape[0], 1))
    normal_train_X_add = np.hstack((add_mar, normal_test_X))
    predict_test_Y = sigmoid(np.dot(normal_train_X_add, theta).flatten())
    for i in range(predict_test_Y.shape[0]):
        if predict_test_Y[i] > 0.5:
            predict_test_Y[i] = 1
        else:
            predict_test_Y[i] = 0
    return predict_test_Y.astype(int), theta

def cv_classify(normal_cv_X, cv_Y, lamb):
    '''
    预测交叉验证集，选取最优k值
    :param lamb：lambda，正则化系数
    :param normal_cv_X: 归一化后的交叉验证集特征
    :param cv_Y：交叉验证集表标签
    :return：最优学习速率，最优迭代次数
    '''
    print("--------------")
    print("Choose Best Alpha and Num_iters:")
    best_alpha = 0
    best_num_iters = 0
    best_score = 0
    alpha_list = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    num_iters_list = [1000, 3000, 10000, 30000, 100000]
    for alpha in alpha_list:
        for num_iters in num_iters_list:
            predict_cv_Y, theta = classify(normal_train_X, train_Y, normal_cv_X, alpha, num_iters, lamb)
            score = accuracy_score(cv_Y,predict_cv_Y)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_num_iters = num_iters
            print("alpha="+str(alpha)+" num_iters="+str(num_iters)+" score="+str(score))
    print("--------------")
    print("best alpha="+str(best_alpha)+" best num iters="+str(best_num_iters)+" best score="+str(best_score))
    return best_alpha, best_num_iters

if __name__ == '__main__':
    train_X, cv_X, train_Y, cv_Y, test_X = load_data()
    normal_train_X, normal_cv_X, normal_test_X = normalization(train_X, cv_X, test_X)

    # 手动实现
    lamb = 1
    alpha, num_iters = cv_classify(normal_cv_X, cv_Y, lamb)
    predict_test_Y, theta = classify(normal_train_X, train_Y, normal_test_X, alpha, num_iters, lamb)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    print("--------------")
    classifier = LogisticRegression()
    classifier.fit(normal_train_X, train_Y)
    score = classifier.score(normal_cv_X, cv_Y)
    predict_cv_Y = classifier.predict(normal_cv_X)
    print("score=" + str(score))
    print("--------------")
    print("Predict Test Dataset:")
    predict_test_Y = classifier.predict(normal_test_X)
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)