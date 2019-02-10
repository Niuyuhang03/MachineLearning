# MachineLearning

机器学习算法实践，参考了[cs 229](https://zh.coursera.org/learn/machine-learning)、[Jack Cui](https://cuijiahua.com/)、[100-Days-Of-ML-Code](https://github.com/MLEveryday/100-Days-Of-ML-Code)等教程提供的相关代码和讲解。请先看每种方法的.md文件，内含方法说明和样例说明。

## 机器学习方法

机器学习算法根据训练集有无标签可以分为有监督学习和无监督学习。有监督学习的训练集中有y值（label），可根据label为数值型还是标称型分为回归和分类。顾名思义，回归问题的label为1.03这样的数值，而分类问题的label多为真假等标签集。无监督学习则没有label，可分为聚类问题和数据降维等。聚类即将样本根据特征分为几类，数据降维则可以直观的显示数据信息。此外还有半监督学习、增强学习等。

机器学习中，我们往往将深度学习单独剥离开来。深度学习指的是使用神经网络的机器学习算法。

集成学习则是将多种机器学习算法结合起来，利用投票等方法选择最终的结果。

## 本项目实现的算法

### 有监督学习：

#### 分类

[kNN](https://github.com/Niuyuhang03/MachineLearning/blob/master/kNN)

[逻辑回归](https://github.com/Niuyuhang03/MachineLearning/blob/master/logistic_regression)

[支持向量机](https://github.com/Niuyuhang03/MachineLearning/blob/master/SVM)

[决策树](https://github.com/Niuyuhang03/MachineLearning/blob/master/decision_tree)：尚未完成

#### 回归

[线性回归](https://github.com/Niuyuhang03/MachineLearning/blob/master/linear_regression)

### 无监督学习

#### 聚类

#### 数据降维

## 数据处理

读取数据：一般从.csv文件读取数据。通过.values函数得到的数据格式为np.array。推荐读出数据后通过.shape函数验证大小。

```python
import pandas as pd
train_dataset = pd.read_csv("filename.csv",header=None)
X = train_dataset.iloc[:, :-1].values
Y = train_dataset.iloc[:, -1].values
# 若csv文件中第一行有标签，还可以这样读取
train_dataset = pd.read_csv("filename.csv")
X = train_dataset.drop(['label'], axis=1).values
Y = train_dataset['label'].values
```
若特征中有male、female等非数值的值，要替换为0和1。注意如果特征中有需要标准化标签时，读出数据时不应加.values。

```python
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])
X = X.values
```

若标签中有多个类别，可以变为独热编码。

```python
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()
```

判断数据中是否有缺失项，如果有，用均值代替。注意isnan函数要求输入为float。

```python
import numpy as np
from sklearn.preprocessing import Imputer
if np.isnan(X.astype(float)).sum() != 0:
    imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit(X)
    X = imp.transform(X).astype(np.int32)
```

拆分训练集和交叉验证集，一般比例为8:2。

```python
from sklearn.model_selection import train_test_split
train_X, cv_X, train_Y, cv_Y = train_test_split(X, Y, test_size=0.2)
```

归一化，将特征缩放到同一范围内。注意只对训练集求缩放的模型，而后应用到训练集、交叉验证集和测试集中。

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

评价：

混淆矩阵：二分类问题时可用

![混淆矩阵](http://ww1.sinaimg.cn/large/96803f81ly1fzf7rkjiqaj20d406oglx.jpg)

```python
# confusion_matrix混淆矩阵。
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cv_Y, predict_cv_Y)
print(cm)
```

准确率：

![准确率](http://ww1.sinaimg.cn/large/96803f81ly1fzf7ucwtzaj207w02rt8m.jpg)

```python
# 准确率
from sklearn.metrics import accuracy_score
score = accuracy_score(cv_Y, predict_cv_Y)
```

F1值：

![F1值](http://ww1.sinaimg.cn/large/96803f81ly1fzf7y11jopj2095034t8q.jpg)

```python
# f1_score， 用于二分类
from sklearn.metrics import f1_score
score = f1_score(cv_Y, predict_cv_Y)
print(cm)
```

均方误差：

![均方误差](http://ww1.sinaimg.cn/large/96803f81ly1fzgd80u212j207a034wed.jpg)

```python
# 均方误差MSE
from sklearn.metrics import mean_squared_error
score = mean_squared_error(cv_Y, predict_cv_Y)
print(score)
```

决定系数r2：

![决定系数](http://ww1.sinaimg.cn/large/96803f81ly1fzgd9y3y8vj206i01ja9y.jpg)

```python
# 决定系数，可用于回归
from sklearn.metrics import r2_score
score = r2_score(cv_Y, predict_cv_Y)
```

将预测结果写入.csv中。

```python
pd.DataFrame(result_new).to_csv('submission.csv', index=False, encoding='utf8', header=False)
```