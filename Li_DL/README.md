# Deep-Learning

李宏毅深度学习系列课程

![DL](Notes/DL_Img/DL_LI.png)

## 学习笔记

**1.Machine Learning ([Notes1](Notes/Notes1_ML.md))**

**2.Deep Learning ([Notes2](Notes/Notes2_DL.md))**

**3.CNN ([Notes3](Notes/Notes3_CNN.md))**

**4.RNN ([Notes4](Notes/Notes4_RNN.md))**

**5.Language Model ([Notes5](Notes/Notes5_LanguageModel.md))**

**7.Adversarial Attack ([Notes7](Notes/Notes7_AdversarialAttack.md))**

## 代码作业

包括李宏毅机器学习课程5次作业

#### HW.1 Regreesion



#### HW.2 Classification

#### HW.3 CNN

#### HW.4 RNN

- **RNN**
  - **任务说明** : 文本情感分类
  - **词向量** : Glove and Word2vec
  - **模型结构** : RNN
  - **额外机制** 
    - ***Unsupervised Learning***
  - **模型效果** 
    - ***测试损失*** : 3.95812
    - ***Accuracy*** : 0.24675

#### HW.8 Seq2Seq

- **Seq2Seq**
  - **任务说明** : 英文翻译中文
  - **词向量** : Word2vec
  - **模型结构** : Encoder-Decoder with Attention
    - ***Encoder*** : Padded Bidirectional LSTM
    - ***Deocder*** : Bidirectional GRU
    - ***Attention*** : Query Attention
  - **额外机制** 
    - ***Scheduler Sampling*** : ratio 0.6
    - ***Beam Search*** : 暂无
  - **模型效果** 
    - ***测试损失*** : 3.95812
    - ***Bleu分数*** : 0.24675

