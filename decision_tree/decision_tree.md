# 决策树

## 简介

有监督分类算法，过程包括决策树生成和剪枝两个过程。决策树如果不经过剪枝过程，很容易过拟合。构建完成后，可以进行可视化，最后一般将模型存储起来。

决策树生成以ID3为例，即从根节点开始，计算每个特征的信息增益，取信息增益最大的特征作为该节点上的特征。信息增益的具体算法可以参考下面这个例子。并不需要记住原始公式，只需记住带入的数字是什么。

![样本](http://ww1.sinaimg.cn/large/96803f81ly1fz7nlhezzyj20nj0hldi3.jpg)

![信息熵](http://ww1.sinaimg.cn/large/96803f81ly1fz7nm51e6oj20no0hn3zk.jpg)

![信息增益](http://ww1.sinaimg.cn/large/96803f81ly1fz7nmh5866j20nm0hmt9o.jpg)

决策树剪枝分为预剪枝和后剪枝两种。所谓剪枝，即判断把该节点及以下节点剪掉，替换为叶节点后，模型的泛化性能是否得到提升。不同的是，预剪枝在生成决策树的每一步都要计算一次，后剪枝则是生成完成后，从最下层的叶节点依次向上计算。

决策树的可视化使用了Matplotlib库，保存则用了pickle.dump函数。

## 实例

[decidion_tree.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/decision_tree/decision_tree.py)：手动搭建参考[jack cui](https://cuijiahua.com/blog/2017/11/ml_2_decision_tree_1.html)文章。