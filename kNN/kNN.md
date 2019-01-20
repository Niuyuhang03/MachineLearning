# kNN

## 简介

k-邻近算法是监督学习的一种分类算法。其核心思想为根据特征，找到距离测试集距离最近的k个训练集，统计训练集的label，取出现次数最多的label作为预测的label。kNN不具有显示学习过程

kNN计算量很大，比较复杂。通常k不大于20，k过小则噪声影响很大，k过大则计算速度较慢，推荐遍历k选取最优值。**kNN通常需要归一化**，使得样本的权重相同。

## 实例

[kNN.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN/kNN.py)：数据集来源为[100-Days-Of-ML-Code](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/datasets/Social_Network_Ads.csv)