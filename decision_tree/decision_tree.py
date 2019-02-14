# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import pickle
from sklearn import tree

def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet:数据集
    :return: 香农熵
    '''
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    划分数据集
    :param dataSet: 待划分的数据集（特征+标签）
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return：划分后的数据集
    '''
    retDataSet = []
    dataSet = np.array(dataSet).tolist()
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return np.array(retDataSet)

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最优特征
    :param dataSet: 训练集（特征+标签）
    :return: 信息增益最大的特征的索引值
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = subDataSet.shape[0] / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    统计classList中出现此处最多的元素
    :param classList: 类标签列表
    :return: 现此处最多的元素
    '''
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels, featLabels):
    '''
    创建决策树
    :param dataSet: 训练数据集（特征+标签）
    :param labels: 特征名称
    :param featLabels: 存储选择的最优特征标签
    :return:
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    dataSet = np.delete(dataSet, bestFeat, 1)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(dataSet, labels, featLabels)
    return myTree

def getNumLeafs(myTree):
    '''
    获取决策树叶子结点的数目
    :param myTree: 决策树
    :return: 决策树的叶子结点的数目
    '''
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取决策树的层数
    :param myTree: 决策树
    :return: 层数
    '''
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    绘制结点
    :param nodeTxt: 结点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 结点格式
    :return:
    '''
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotMidText(cntrPt, parentPt, txtString):
    '''
    标注有向边属性值
    :param cntrPt: 用于计算标注位置
    :param parentPt: 用于计算标注位置
    :param txtString: 标注的内容
    '''
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    '''
    绘制决策树
    :param myTree: 决策树
    :param parentPt: 标注的内容
    :param nodeTxt: 结点名
    '''
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    '''
    创建绘制面板
    :param inTree: 决策树
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def classify(inputTree, featLabels, testVec):
    '''
    预测
    :param inputTree: 已经生成的决策树
    :param featLabels: 存储选择的最优特征标签
    :param testVec: 测试数据列表，顺序对应最优特征标签
    :return: 分类结果
    '''
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    '''
    存储决策树
    :param inputTree: 决策树
    :param filename: 决策树的存储文件名
    '''
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    """
    函数说明:读取决策树

    Parameters:
        filename - 决策树的存储文件名
    Returns:
        pickle.load(fr) - 决策树字典
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)

def load_data():
    '''
    读入训练集和测试集
    :return:训练集特征、交叉验证集特征、训练集标签、交叉验证集标签、测试集特征
    '''
    train_data = pd.read_csv('train.csv', header=None)
    X = train_data.iloc[:, 1:4].values
    Y = train_data.iloc[:, 4].values
    # 将缺失数据替换为平均值
    if np.isnan(X.astype(float)).sum() > 0:
        print("NaN exists in train_X.")
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(X)
        X = imp.transform(X).astype(np.int32)
    # 划分训练集和交叉验证集，比例为8:2
    train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)

    test_data = pd.read_csv('test.csv', header=None)
    test_X = test_data.iloc[:, 1:].values
    return train_X, cv_X, train_Y, cv_Y, test_X

def Classify_tree(normal_train_X, train_Y, normal_test_X):
    '''
    预测
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_test_X: 归一化后的测试集特征
    :return: 测试集标签
    '''
    num_normal_train_X = normal_train_X.shape[0]
    num_normal_test_X = normal_test_X.shape[0]
    predict_test_Y = []
    featLabels = []
    dataset = np.append(normal_train_X, train_Y.reshape((num_normal_train_X,1)), axis=1)
    labels = ['Gender','Age','EstimatedSalary']
    myTree = createTree(dataset, labels, featLabels)
    storeTree(myTree, 'classifierStorage.txt')
    createPlot(myTree)
    for i in range(num_normal_test_X):
        result = classify(myTree, featLabels, normal_test_X[i, :])
        predict_test_Y.append(result)
    return np.array(predict_test_Y)

def cv_Classify_tree(normal_train_X, train_Y, normal_cv_X, cv_Y):
    '''
    预测交叉验证集，选取最优k值
    :param normal_train_X: 归一化后的训练集特征
    :param train_Y: 训练集标签
    :param normal_cv_X: 归一化后的交叉验证集特征
    :param cv_Y：交叉验证集表标签
    '''
    print("--------------")
    print("Test CV:")
    predict_cv_Y = Classify_tree(normal_train_X, train_Y, normal_cv_X)
    score = f1_score(cv_Y, predict_cv_Y)
    print(" f1_score="+str(score))
    cm = confusion_matrix(cv_Y, predict_cv_Y)
    print(cm)

if __name__ == '__main__':
    train_X, cv_X, train_Y, cv_Y, test_X = load_data()

    # 手动实现
    cv_Classify_tree(train_X, train_Y, cv_X, cv_Y)
    predict_test_Y = Classify_tree(train_X, train_Y, test_X)
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)

    # 调库
    dec_tree = tree.DecisionTreeClassifier(criterion='gini')
    dec_tree.fit(train_X, train_Y)
    predict_test_Y = dec_tree.predict(test_X)
    print(dec_tree.score(cv_X, cv_Y))
    print("--------------")
    print("Predict Test Dataset:")
    print(predict_test_Y)
    pd.DataFrame(predict_test_Y).to_csv('submission.csv', index=False, encoding='utf8', header=False)