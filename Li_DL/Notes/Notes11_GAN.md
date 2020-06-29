# Generative Adversarial Network

## Basic idea of GAN

### Generator

![11_1](DL_Img/notes11/11_1.png)

**Conditional Generation** : Will control what to generate latter. 

![11_9](DL_Img/notes11/11_9.png)

### Discriminator

![11_2](DL_Img/notes11/11_2.png)

**Discriminator —— Teacher, Generator —— Student.**

Discriminator tell the Generator where is not good enough.

![11_10](DL_Img/notes11/11_10.png)

### Train Algorithm

- **Algorithm:**
  - Initilaize generator and discriminator
  - In each training iteration:
    - **Fix Generator G, and update discriminator D**
      - Discriminator learns to assigh high scores to real projects and low scores to generator objects.
    - **Fix discriminator D, and update Generator G**
      - Generator learns to "fool" the discriminator

This is where the term **'adversarial'** comes from.

![11_3](DL_Img/notes11/11_3.png)

## GAN as Structured Learning

### Structured Learning

![11_4](DL_Img/notes11/11_4.png)

- **Output:**
  - **Output Sequence :**
    - ***Machine Trainslation*** : X - sentence of language 1 ; Y - sentence of language 2
    - ***Speech Recognition*** : X - speech ; Y - transcription
    - ***Chat-bot*** : X - what a user says ; Y - response of machine
  - **Output Image:**
    - ***Image to Image*** : X - input image ; Y - outpout image
    - ***Text to Image*** : X - input text ; Y - output image

### Why structure Learning Challenging ?

![11_5](DL_Img/notes11/11_5.png)

![11_6](DL_Img/notes11/11_6.png)

### Structured Learning Approach

***Bottom Up*** : **Generator** - Learn to generate the object at the component level.

***Top Down*** : **Discriminator** - Evaluating the whole object, and find the best one.

## Can Generator learn by itself ?

The relation between the components are critical. Although highly correlated, the cannot influence each other.

Need deep structure to catch the relation between components.

## Can Discriminator generate ?

Discriminator can used to generate.But Discriminator training needs some negative examples.

Discriminator needs general algorithm to generate negative examples.

![11_7](DL_Img/notes11/11_7.png)

## Benefit of GAN

![11_8](DL_Img/notes11/11_8.png)