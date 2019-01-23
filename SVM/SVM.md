# 支持向量机

## 简介

SVM是有监督的分类算法。将所有样本投射到一个n维空间中，在空间中找到一条超平面，且SVM要求这条超平面距离两侧样本都最远。

![超平面](http://ww1.sinaimg.cn/large/96803f81ly1fzgefaiockj209d0bbjsf.jpg)

同逻辑回归一样，SVM直接使用时只能用来二分类。但SVM泛化错误率低，计算开销小，可以说是最好的现成的分类器。

## 实例

[SVM.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/SVM/SVM.py)