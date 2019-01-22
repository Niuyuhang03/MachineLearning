# 逻辑回归

## 简介

逻辑回归（LR）是有监督的分类算法，一般的逻辑回归只能用于二分类，多分类需要多个逻辑回归模型。其原理和线性回归类似，但给回归方程加上了sigmoid函数，使得结果映射为0和1。相应的，修改代价函数，梯度下降求出最优θ。逻辑回归实质上是找到一个决策边界。

逻辑回归的回归函数一般使用sigmoid函数：

![sigmoid](http://ww1.sinaimg.cn/large/96803f81ly1fzfkiptauxj20dv0dtq3f.jpg)

可以看到，S函数将在x>0时，y>0.5，在x<0时，y<0.5，因此可以将测试集标签分为1和0。

逻辑回归的代价函数为：

![代价函数](http://ww1.sinaimg.cn/large/96803f81ly1fzfl0e1qlvj20rv03o74i.jpg)

梯度：

![梯度](http://ww1.sinaimg.cn/large/96803f81ly1fzfn4boo3yj20gk03ut8x.jpg)

而预测的函数为：

![预测](http://ww1.sinaimg.cn/large/96803f81ly1fzfl5ml58cj20gl01s3yj.jpg)

逻辑回归需要用正则化，利用正则化系数lambda作为惩罚，防止过拟合。

正则化后的梯度：

![梯度](http://ww1.sinaimg.cn/large/96803f81ly1fzfojo2wiej20q406lgmm.jpg)

## 实例

[logistic_regression.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/logistic_regression/logistic_regression.py)