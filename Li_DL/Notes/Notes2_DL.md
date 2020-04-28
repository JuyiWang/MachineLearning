# Notes.2 Deep Learning

## Three Steps

### 1. Define a function set

***Give network structure,define a function set***

**Deep → Many hidden layers**

Hidden layer : Feature extractor, replacing feature engineering.

***Decide the network structure to find the best function***

### 2. Goodness of function

![DL-1](DL_Img/DL_1.png)

### 3. Pick the best function

**Backpropogation** : Compute gradient efficiently

BP算法本质上还是Gradient Descent.

## Why Deep Learning?

### 1. Fat & Short Vs Thin & Tall

![DL-2](DL_Img/DL_2.png)

### 2. Modularizaton

**Deep → Modularization**

![DL_3](DL_Img/DL_3.png)

***Shallow network can represent any function, but using deep structure is more effective.***

相比 Fat Model,Deep Model 需要更少的数据进行训练。并且通过 Modularization 可自动从数据中学习参数（**End-to-End training**）,可减少特征工程量和人工操作。

***Less engineering labor, but machine learns more.***

## BackPropagation

**Backpropagation : Gradient Descent + Chain Rule**

### Chain Rule

![DL_4](DL_Img/DL_4.png)

### Forward pass & Back pass

![DL_5](DL_Img/DL_5.png)

$\frac{\partial C}{\partial w} = \frac{\partial z}{\partial w} \frac{\partial C}{\partial z}$

**Forward psaa:** Compute $\frac{\partial z}{\partial w}$ for all **parameters**
 
**Back pass:** Compute $\frac{\partial C}{\partial z}$ for all **activation function inputs z**

**1. Forward pass**

Forward pass 本质为一阶求导。 $\frac{\partial z}{\partial w_i} = x_i$ 所得值为与 $w$ 相连的输入值。

**2. Back pass**

![DL-6](DL_Img/DL_6.png)

Back pass 的计算过程实际是应用链式求导法则计算计算反向梯度的过程。可将这个过程看做一个反向的Neural Network进行计算。

![DL_8](DL_Img/DL_8.png)

其中$ a = \sigma(z)$ , $\sigma(z)$ 为 Sigmoid Function。

![DL-7](DL_Img/DL_7.png)

对于每一个 **activation function** 的输入 $z$ ,应用 Back pass 直至 $C$ 为输出层的误差。

## Tips for Deep Learning
