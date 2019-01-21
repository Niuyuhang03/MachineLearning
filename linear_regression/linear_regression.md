# 线性回归

## 简介

线性回归是一种有监督的回归算法，目的是已知X和Y，通过拟合出回归系数θ，得到回归方程进行预测。

拟合回归系数θ的过程，实际上就是求使得预测的误差（平方误差）最小时的θ，即对平方误差求导，导数为0时的θ。线性回归实际有正规方程和梯度下降两种求解方法，后者可以通过归一化加快迭代速度。正规方程即直接求出导数为0时的θ值：

![回归系数](http://ww1.sinaimg.cn/large/96803f81ly1fze1s7vej2j206a00y0sk.jpg)

然而矩阵有逆要求为非奇异矩阵，且这个方法在矩阵X较大时速度慢于梯度下降，因此常用梯度下降法求解线性回归。

定义成本函数：

![成本函数](http://ww1.sinaimg.cn/large/96803f81ly1fze1zdopk1j20bj03fmx5.jpg)

求得成本函数的最小值应该在梯度为0时得到。

梯度：

![梯度](http://ww1.sinaimg.cn/large/96803f81ly1fze21hsecrj202p02cglf.jpg)

因此，迭代如下公式更新θ直至收敛，即可得到回归系数θ。注意更新时，要求在一轮内，用上一个θ计算完所有项后，再更新新的θ。

![迭代更新θ](http://ww1.sinaimg.cn/large/96803f81ly1fze2286f11j209702ndfr.jpg)

## 实例

[linear_regression.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/linear_regression/linear_regression.py)