## 第一周笔记 Week_1

### 一、引言

#### 1.1 机器学习

Tom Mitchell :一个计算机程序能够从经验E中，解决任务T，达到性能度量P。有了经验E，经过度量P评判，程序在处理任务T的时候表现提升。

算法分类：监督学习、无监督学习、强化学习、推荐系统等。

问题分类：回归问题 Regression Problem | 分类问题 Classification Problem

#### 1.2 监督学习

监督学习：Supervised Learning

给定算法数据集，数据集中包含正确的Label。例：房价预测、肿瘤类型判断、邮件分类。

#### 1.3 无监督学习

无监督学习：Unspervised Learning

无监督学习中定的数据集中，不包含任何Label。例：聚类，谷歌新闻、鸡尾酒宴。

### 二、单变量线性回归(Univariate Linear Regession)

#### 2.1 模型表示

![hypothesis](Img/../../../Img/Hypothesis.png)

学习算法( Learning Algorithm )的作用：根据给定的训练集，输出一个假设函数( Hypothesis function : $h$ )。假设函数实际是输入$x$与输出标签$y$之间的函数映射。$h$根据输入的$x$计算$y$。

单变量线性回归假设函数：$h_\theta(x) = \theta_0 + \theta_1x$。对于给定的$\theta$，$h_\theta(x)$为$x$的函数。

#### 2.2 代价函数

代价函数（Cost Function）：如何去选择$\theta$？

针对房价预测问题，模型所预测的值与训练集中实际值之间的差距（下图中蓝线所指）就是建模误差（modeling error）。

![Cost](Img/../../../Img/Cost.png)

使用均方误差（MSE）作为代价函数。

![J](Img/../../../Img/J.png)

目标是选择出可以使得建模误差的平方和最小的模型参数：

$minimize_{\theta_0,\theta_1}J(\theta_0,\theta_1)$

#### 2.3 梯度下降

针对已经获得的代价函数$J(\theta_0,\theta_1)$,希望得到使其取得最小值的参数$\theta_0$，$\theta_1$。

**操作步骤：**

1. 设置初始$\theta_0$、$\theta_1$,计算$J(\theta)$
2. 在达到所设阈值前改变$\theta_0$、$\theta_1$减小$J(\theta)$的值。
   
梯度下降算法每次找到的都是局部最优解，取决于初始参数组合，不一定为全局最优解。

![Gradient descent](Img/../../../Img/Gd.jpg)

批量梯度下降（batch gradient descent）算法的公式为：

![Batch Gradient Descent](Img/../../../Img/GD-2.png)

$\alpha$是学习率（Learning rate），决定沿着能让代价函数下降程度最大的方向向下迈出的step有多大。

如果学习率$\alpha$过小，那么梯度更新会非常缓慢。如果学习率$\alpha$过大，可能会越过最低点，导致无法收敛甚至发散。每一步参数都更接近最低点，代价函数的导数也逐渐减小，最小值时为0，因此即便$\alpha$不变也可以收敛至最小值。

![Gradient descent](Img/../../../Img/GD-3.png)

在批量梯度下降中，每一次都同时让所有的参数减去学习速率乘以代价函数的导数。在梯度更新过程中，所有的参数必须同时更新，每一步计算的$J(\theta)$都是关于所有$\theta$的函数，如果异步更新会导致每次计算的$J(\theta)$发生变化。

### 三、多变量线性回归(Linear Regression)

#### 3.1 多维特征

#### 3.2 多变量梯度下降

#### 3.3 梯度下降实践-特征缩放

#### 3.4 梯度下降实践-学习率

#### 3.5 特征和多项式回归

#### 3.6 正规方程

### 四、逻辑回归(Logistic Regression)

#### 4.1 分类问题

#### 4.2 假说表示、判定边界

#### 4.3 代价函数和梯度下降

#### 4.4 高级优化

### 五、正则化

#### 5.1 欠拟合与过拟合

#### 5.2 代价函数

#### 5.3 正则化线性回归

#### 5.4 正则化的逻辑回归模型