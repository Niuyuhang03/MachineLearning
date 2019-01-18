# MachineLearning

机器学习算法实践，参考了[Jack Cui](https://cuijiahua.com/)等教程提供的相关代码和讲解。请先看每种方法的.md文件，内含方法说明和样例说明。

样本数据来自[Jack Cui](https://github.com/Jack-Cherish/Machine-Learning)，[北航数据工作站](http://contest.mooc.buaa.edu.cn/)等网站，以及《机器学习实践》中的内容。

## 机器学习方法

机器学习算法根据训练集有无标签可以分为有监督学习和无监督学习。有监督学习的训练集中有y值（label），可根据label为数值型还是标称型分为回归和分类。顾名思义，回归问题的label为1.03这样的数，而分类问题的label多为真假等标签集。而无监督学习则没有label，可分为聚类问题和数据降维等。聚类即将样本根据特征分为几类，数据降维则可以直观的显示数据信息你。此外还有半监督学习。

机器学习中，我们往往将深度学习单独剥离开来。深度学习指的是使用神经网络的机器学习算法。

集成学习则是将多种机器学习算法结合起来，利用投票等方法选择最终的结果。

## 本项目实现的算法

### 有监督学习：

#### 分类

[kNN](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN)：kNN实现时，采用了对测试集进行纵向复制、相减后求距离，排序时返回索引，统计label时用了字典的get函数，归一化时采用.max(0)函数，大大简化了代码。

[决策树](https://github.com/Niuyuhang03/MachineLearning/blob/master/decision_tree)：实现了决策树的可视化、保存等工作

#### 回归

### 无监督学习

#### 聚类

#### 数据降维

## 数据处理

一般通过.csv文件读取数据。通过.values函数得到的数据格式为np.array。读出数据通过.shape函数验证大小。

```python
dataset = pd.read_csv("filename.csv")
X = dataset.drop(['label'], axis=1).values
Y = dataset['label'].values
# 无标签则用如下方式读数据
# dataset = pd.read_csv("filename.csv",header=None)
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values
```

判断数据中是否有缺失项，如果有，用均值代替。

```python
from sklearn.preprocessing import Imputer
if np.isnan(X).sum() != 0:
    imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit(X)
    X = imp.transform(X)
```

拆分训练集和交叉验证集。

```python
from sklearn.model_selection import train_test_split
train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)
```

归一化，注意只对训练集求均值方差，而后应用到训练集、交叉验证集和测试集中。

```python
# 均值方差归一化，多用于距离相关的模型，如K-means
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
cv_X = sc_X.transform(cv_X)
test_X = sc_X.transform(test_X)
```

```python
# 线性归一化，多用于图片等
from sklearn.preprocessing import MinMaxScaler
mc_X = MinMaxScaler()
train_X = mc_X.fit_transform(train_X)
cv_X = mc_X.transform(cv_X)
test_X = mc_X.transform(test_X)
```

写入.csv中。

```python
pd.DataFrame(result_new).to_csv('submission.csv', index=False, encoding='utf8', header=False)
```