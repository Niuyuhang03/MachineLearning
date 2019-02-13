# k-means

## 简介

k-means是非监督学习的一种分类算法。通过选取k个样本作为初始聚类中心，计算其他样本到各个聚类中心的距离，将所有样本划分给最近的一个聚类中心，再在每个聚类中求出中心值作为新的聚类中心，重新分配，直至样本不再移动。

k-means收敛较慢，且可能会收敛到局部最小值。可以通过误差平方和SSE来求出最优k值。

## 实例

[k-means.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/k-means/k-means.py)：数据集来源为[100-Days-Of-ML-Code](https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/datasets/Social_Network_Ads.csv)