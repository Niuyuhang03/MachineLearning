import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def read_train_data():
    '''
    读入训练集
    :return:训练集特征、交叉验证集特征、训练集标签、交叉验证集标签
    '''
    train_data = pd.read_csv('train.csv')
    X = train_data.iloc[:, 1:4].values
    Y = train_data.iloc[:, 4].values
    # 将male和female转为1和0
    labelencoder_X = LabelEncoder()
    X[:,0] = labelencoder_X.fit_transform(X[:,0])
    # 将缺失数据替换为平均值
    if np.isnan(X.astype(float)).sum() > 0:
        print("NaN exists in train_X.")
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(X)
        X = imp.transform(X).astype(np.int32)
    # 划分训练集和交叉验证集，比例为8:2
    train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)
    return train_X, cv_X, train_Y, cv_Y

def read_test_data():
    '''
    读入测试集
    :return:测试集特征
    '''
    test_data = pd.read_csv('test.csv')
    test_X = test_data.iloc[:, 1:4].values
    # 将male和female转为1和0
    labelencoder_test_X = LabelEncoder()
    test_X[:, 0] = labelencoder_test_X.fit_transform(test_X[:, 0])
    return test_X

def normalization(train_X, cv_X, test_X):
    '''
    归一化
    :param train_X: 训练集特征
    :param cv_X: 交叉验证集特征
    :return: 归一化后的训练集特征和交叉验证集特征
    '''
    sc_X = StandardScaler()
    normal_train_X = sc_X.fit_transform(train_X)
    normal_cv_X = sc_X.transform(cv_X)
    normal_test_X = sc_X.transform(test_X)
    return normal_train_X, normal_cv_X, normal_test_X

def Classify(normal_train_X, train_Y, normal_test_X, k):
    '''
    预测
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_test_X: 归一化后的测试集特征
    :param k: 最近k个点
    :return: 预测集标签
    '''
    num_normal_train_X = normal_train_X.shape[0]
    num_normal_test_X = normal_test_X.shape[0]
    predict_test_Y = []
    for i in range(num_normal_test_X):
        # 赋值测试集每一行，与训练集由欧式距离求出距离后排序
        sq_diff = (np.tile(normal_test_X[i, :], (num_normal_train_X, 1)) - normal_train_X) ** 2
        diff = (sq_diff.sum(axis=1)) ** 0.5
        # 使用排序后的索引
        sorted_diff_index = diff.argsort()
        predict_label = {}
        for j in range(k):
            label = train_Y[sorted_diff_index[j]]
            predict_label[label] = predict_label.get(label, 0) + 1
        sorted_predict_label = sorted(predict_label.items(), key=operator.itemgetter(1), reverse=True)
        predict_test_Y.append(sorted_predict_label[0][0])
    return np.array(predict_test_Y)

def cv_Classify(normal_train_X, train_Y, normal_cv_X, cv_Y):
    '''
    预测交叉验证集，选取最优k值
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_cv_X: 归一化后的交叉验证集特征
    :param cv_Y：交叉验证集表标签
    :return: 最优k值
    '''
    print("--------------")
    print("Choosing Best K:")
    best_k = 1
    best_f1_score = 0
    best_predict_cv_Y = []
    for k in range(1, 21):
        predict_cv_Y = Classify(normal_train_X, train_Y, normal_cv_X, k)
        score = f1_score(cv_Y, predict_cv_Y)
        if score > best_f1_score:
            best_k = k
            best_f1_score = score
            best_predict_cv_Y = predict_cv_Y
        print("k="+str(k)+" f1_score="+str(score))
    print("--------------")
    print("Score in Best K:")
    cm = confusion_matrix(cv_Y, best_predict_cv_Y)
    print(cm)
    print("best k="+str(best_k)+" best f1_score="+str(best_f1_score))
    return best_k

if __name__ == '__main__':
    train_X, cv_X, train_Y, cv_Y = read_train_data()
    test_X = read_test_data()
    normal_train_X, normal_cv_X, normal_test_X = normalization(train_X, cv_X, test_X)

    # 手动实现
    best_k = cv_Classify(normal_train_X, train_Y, normal_cv_X, cv_Y)
    predict_test_Y = Classify(normal_train_X, train_Y, normal_test_X, best_k)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    print("--------------")
    print("Choosing Best K:")
    best_k = 1
    best_f1_score = 0
    best_predict_cv_Y = []
    for k in range(1, 21):
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(normal_train_X, train_Y)
        score = classifier.score(normal_cv_X, cv_Y)
        if score > best_f1_score:
            best_k = k
            best_f1_score = score
            best_predict_cv_Y = classifier.predict(normal_cv_X)
        print("k=" + str(k) + " f1_score=" + str(score))
    print("--------------")
    print("Score in Best K:")
    cm = confusion_matrix(cv_Y, best_predict_cv_Y)
    print(cm)
    print("best k=" + str(best_k) + " best f1_score=" + str(best_f1_score))
    print("--------------")
    print("Predict Test Dataset:")
    classifier = KNeighborsClassifier(n_neighbors=best_k)
    classifier.fit(normal_train_X, train_Y)
    predict_test_Y = classifier.predict(normal_test_X)
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)