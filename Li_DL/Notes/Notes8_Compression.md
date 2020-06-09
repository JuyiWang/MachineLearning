# Network Compression

## Network Pruning

### Pruning 

![network pruning](DL_Img/notes8/8_1.png)

相比于大的模型，小模型难以优化。因此选择模型压缩而不是直接训练一个较小的模型。

#### Pruning Weights

![Weights pruning](DL_Img/notes8/8_2.png)

#### Pruning Neuron

![Neuron pruning](DL_Img/notes8/8_3.png)

## Knowledge Distillation

用小模型学习大模型学得的知识。

![KD](DL_Img/notes8/8_4.png)

## Parameter Quantization

![Parameter Quntization](DL_Img/notes8/8_5.png)

## Architecture Design

![AD](DL_Img/notes8/8_6.png)

通过更改网络结构减少参数量。

![AD](DL_Img/notes8/8_7.png)

![AD](DL_Img/notes8/8_8.png)

- **Depthwise Convolution**
  - **Filter number = Input channel number.**
  - **Each filter only consider one channel.**
  - **The filters are K*K matrices.**
  - **There is no interaction between channels.**

![AD](DL_Img/notes8/8_9.png)

## Dynamic Computation

![DC](DL_Img/notes8/8_10.png)

提高网络中间层的预测能力。