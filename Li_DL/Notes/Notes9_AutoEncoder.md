# Auto-Encoder



### Auto-Encoder
![9-1](DL_Img/notes9/9_1.png)

![9-2](DL_Img/notes9/9_2.png)

Output of the hidden layer is the code. Called Bottleneck later.
### Text Retrieval

![9-3](DL_Img/notes9/9_3.png)

Semantics are not considered in Bag-of-word.

### Similiar Image Search

Retrieved using Euclidean distance in pixel intensity space.

![9-4](DL_Img/notes9/9_4.png)

### Auto-Encoder for CNN

![9_5](DL_Img/notes9/9_5.png)

Auto-Encoder for CNN includes : **Deconvolution and Unpooling**

#### Deconvolution

**Deconvolution is similiar with convolution.**

![9_6](DL_Img/notes9/9_7.png)

#### Unpooling

![9_7](DL_Img/notes9/9_6.png)

**Unpooling with the Max Locations Informations.**

### Pre-training DNN

![9_8](DL_Img/notes9/9_8.png)

$W_1$ is the parameters that wants to be initialize.

Pre-training DNN 在含有大量无标签数据时可用。

当 input_size 小于hidden_size 时，如上图 784—1000，要对 $W_1$ 添加较大的正则化系数，以防将 input 复制进 hidden, 造成 hidden 稀疏。

### Good embedding ？

An embedding should represent the object.

![9_9](DL_Img/notes9/9_9.png)

### Sequential Data

![9_10](DL_Img/notes9/9_10.png)

![9_11](DL_Img/notes9/9_11.png)

例：文本摘要

### Feature Disentangle

An object contains multilpe aspect information.The same sentence has different impact when it is said by different people.

- 一段语音：
  - 语音信息
  - 语者信息
- 一段文字
  - 语法信息
  - 语义信息

![9_12](DL_Img/notes9/9_12.png)

Such as : **Voice Conversion , Adversarial Training , Designed Network Architecture**

![9_13](DL_Img/notes9/9_13.png)

### Discrete Representation

Let embedding easier to interpret or clustering.Such as **One-hot vector** and **Binary Vector**.

![9_14](DL_Img/notes9/9_14.png)

Let Discriminatorc consider the not readable summary as real

![9_15](DL_Img/notes9/9_15.png)