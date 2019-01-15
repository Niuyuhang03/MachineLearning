# kNN

## 简介

k-邻近算法是监督学习的一种分类算法。其核心思想为根据特征，找到距离测试集距离最近的k个训练集，统计训练集的label，取出现次数最多的label作为预测的label。

kNN计算量很大，比较复杂。通常k不大于20，KNN不具有显示学习过程。**kNN通常需要归一化**，使得样本的权重相同。

## 实例

1. 根据电影中的接吻镜头数和打斗镜头数预测电影为爱情电影还是动作电影，无数据集
   [filmPredict.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN/filmPredict.py)

2. 根据约会网站指标选则心仪程度，**数据集为txt文件，做了数据可视化、归一化、正确率验证器**[datingPredict.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN/datingPredict.py)

3. 手写数字识别，**调用sklearn库**[digitRecognize.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN/digitRecognize.py)