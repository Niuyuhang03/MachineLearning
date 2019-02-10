# 支持向量机

## 简介

SVM是有监督的分类算法。将所有样本投射到一个n维空间中，在空间中找到一条超平面，且SVM要求这条超平面距离两侧样本都最远。

![超平面](http://ww1.sinaimg.cn/large/96803f81ly1fzgefaiockj209d0bbjsf.jpg)

同逻辑回归一样，SVM直接使用时只能用来二分类。但SVM泛化错误率低，计算开销小，可以说是最好的现成的分类器。

SVM用序列最小优化算法SMO可以较快求解，SMO主要有如下步骤：求解误差，计算上下界，计算学习速率，更新乘子αi，修剪αi，更新αi，更新b1和b2，更新b。

## 实例

[SVM.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/SVM/SVM.py)，手动搭建参考[jack cui](https://cuijiahua.com/blog/2017/11/ml_8_svm_1.html)文章。