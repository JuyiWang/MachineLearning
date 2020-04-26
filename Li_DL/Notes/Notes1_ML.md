# Notes.1 ML

## Machine Learning

### 优化目标

**1. 学习目标**

***What we really care about is the error on new data.***

以在训练集上的期望训练风险代替在真实数据上的期望泛化误差。基于最小化平均训练误差的训练过程称为 *经验风险最小化*。

**2. 学习过程**

*1. Design Model*

*2. Goodness of Function*

在机器学习问题中，关注的某些性能度量 $P$ ，如分类问题中的Accuracy，Recall，F1-Score，很大可能都是在测试集上不可解的。因此通过定义LossFunction $J(\theta)$ 来提高 $P$ 。

***Input a function ,ouput how bad it is.*** LossFunction的作用就是衡量参数 $\theta$ 的好坏。

*3. Gradient Descent*

**3. Smooth function**

***Regularization*** 可避免过拟合（Overfitting） $L(\theta) = J(\theta) + \lambda \sum (\theta)^2$

当测试集中存在noise data时，smooth function 收到噪声输入的影响较小。

***Prefer smooth function bue not too smooth.***

***Gradient*** 的方向为 loss 等高线的法线方向。

## Regression
