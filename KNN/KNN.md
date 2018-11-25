# KNN

## [简介](http://cuijiahua.com/blog/2017/11/ml_1_knn.html)

k-邻近算法是监督学习的一种分类算法，核心思想为找到测试集距离最近的k个训练集，统计训练集的label，取出现次数最多的label作为预测的label。通常k不大于20，KNN不具有显示学习过程。

## 实例

1. 根据电影中的接吻镜头数和打斗镜头数预测电影为爱情电影还是动作电影，**无数据集**，简单练习
   [filmPredict.py](filmPredict.py)

2. 根据约会网站指标选则心仪程度，**数据集为txt文件，做了数据可视化、归一化、正确率验证器**[datingPredict.py](datingPredict.py)

3. 手写数字识别，**调用sklearn库**[digitRecognize.py](digitRecognize.py)