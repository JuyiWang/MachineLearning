# Transfer Learning

Data **not directly realted to** the task considered.

Similar doamin, different tasks or Different domains, same task.

## Overview

![12_1](DL_Img/notes12/12_1.png)

**Different terminology in different literature.**

## Source labelled & Target unlabelled

### Model Fine-tuning

![12_2](DL_Img/notes12/12_2.png)

#### Conservatice Training

![12_3](DL_Img/notes12/12_3.png)

#### Layer Transfer

![12_4](DL_Img/notes12/12_4.png)

Use source data train an multi-layer model, use target data train uncopy layer.

- **Which layer can be transferred ?**
  - **Speech** : usually copy the **last** few layers
  - **Image** : usually copy the **first** few layers

### Multitask Learning

![12_5](DL_Img/notes12/12_5.png)

## Source labelled & Target unlabeled

### Domain-adversarial training

![12_6](DL_Img/notes12/12_6.png)

Similar to GAN, but domain classifier is easy to "fool".

Domain-adversarial want both of label predictor and domain classifier performance well.

**Use *[ gradient reversal layer ]* between domain classifier and feature extractor, let the feature extractor ignore the domain information.**

***[ gradient reversal layer ]*** : $new gradient desent = gradient desent * (-1) $, Opposite direction of gradient

### Zero-shot learning

![12_7](DL_Img/notes12/12_7.png)

![12_8](DL_Img/notes12/12_8.png)

![12_9](DL_Img/notes12/12_9.png)

![12_11](DL_Img/notes12/12_11.png)

**What if we don't have database ? Attribute embedding + word embedding.**

![12_10](DL_Img/notes12/12_10.png)

## Source unlabeled & Target labelled

### Self-taught learning

Learning to extract better representation from the source data(unsupervised approach)

Extracting better representation for target data