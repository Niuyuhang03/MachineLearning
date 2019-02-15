# ANN

## 简介

人工神经网络ANN是最简单的深度学习算法之一。其基本结构如下：

![ANN](http://ww1.sinaimg.cn/large/96803f81ly1g06zla4re8j20if09cq73.jpg)

模型包含一个输入层，若干个隐藏层和一个输出层。隐藏层越多效果越好，训练难度也越大，全连接神经网络隐藏层一般不超过三层。每层有若干个神经元，隐藏层神经元个数可以用sqrt(n*l)来作为初始值，其中n和l为输入层和输出层的神经元个数。神经元之间采用全连接，连接上有权重w。当要计算下一层某个神经元的值时，所用公式为：

![y1](http://ww1.sinaimg.cn/large/96803f81ly1g06zt0vxr7j207a01aglg.jpg)

即每个要计算的神经元的值等于上一层与它连接的所有神经元与连接上的权重的乘积和，输入到sigmoid的结果。其中sigmoid函数为：

![sigmoid](http://ww1.sinaimg.cn/large/96803f81ly1g06ztlqf7tj209o029jr8.jpg)

则模型的计算用向量来表示为：

![向量](http://ww1.sinaimg.cn/large/96803f81ly1g06zs4fji5j20pi05zdgx.jpg)

其中每层权重有一个偏置项Wb值为1。相应的，内层神经元也会多一个和偏置项相乘的项。

![计算](http://ww1.sinaimg.cn/large/96803f81ly1g06zrph176j205w01fwea.jpg)

其中的f函数就是sigmoid。因此对于有三个隐藏层的神经网络：

![ANN](http://ww1.sinaimg.cn/large/96803f81ly1g06zwoeyr4j20im09vad5.jpg)

每一层的结果可表示为：

![结果](http://ww1.sinaimg.cn/large/96803f81ly1g06zxf183tj206g04caaa.jpg)

所以只需要计算出权重即可，权重由反向传播算法得到：

对于输出层，求得误差：

![输出层误差](http://ww1.sinaimg.cn/large/96803f81ly1g0706wi1yhj207f01r744.jpg)

对于隐藏层，求得误差：

![隐藏层误差](http://ww1.sinaimg.cn/large/96803f81ly1g0707ntjtqj20ah01ma9x.jpg)

更新权重：

![更新权重](http://ww1.sinaimg.cn/large/96803f81ly1g07086pqqyj206s01la9v.jpg)

其中η是学习速率。注意偏置项也需要更新。

## 实例

[ANN.py](https://github.com/Niuyuhang03/MachineLearning/blob/master/ANN/ANN.py)：数据集采用[MINIST](http://yann.lecun.com/exdb/mnist)