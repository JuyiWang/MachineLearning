# Meta Learning

Learn to Learn

![13_1](DL_Img/notes13/13_1.png)


![13_2](/Li_DL/Notes/DL_Img/notes13/13_2.png)

**Machine Learning : The ability to find a function based on data.**

**Meta Learning : The alility to a function that finding a function based on data.**

## Three Steps

- **Meta Learning is Simple**:
  - Define a set of Function f
  - Goodness of Function f
  - Pick the best Function f

**Function f** : the Learning algorithm to find function

### Define a set of learning algorithm

![13_3](/Li_DL/Notes/DL_Img/notes13/13_3.png)

Replace humans with machines to define **Initialize parameters $\theta$** or **Gradient Update**.

### Define goodness of a funcation F

![13_6](/Li_DL/Notes/DL_Img/notes13/13_6.png)

Using actual task to evaluate the generated model. Multi task multi test.

$L(F) = \sum_{n=1}^{N} l^{n}$ 

$N$ : N tasks.

$l^{n}$ : Testing loss for task n after training.

**Few-shot learning** : Small amount of label data

### Pick the best function F

![13_4](/Li_DL/Notes/DL_Img/notes13/13_4.png)

### Bench mark

![13_5](/Li_DL/Notes/DL_Img/notes13/13_5.png)

**Few-shot Classification** : Make sure that the training data and the test data don't overlap.


## MAML & Reptile

### MAML

**Only focus on initialization parameter.**

#### MAML v.s. Pre-Training

**Loss function is different.**

![13_7](/Li_DL/Notes/DL_Img/notes13/13_7.png)

**Model Pre_training** : Train the model in the task with large dataset, and fine-tune in the task with small dataset.

![13_8](/Li_DL/Notes/DL_Img/notes13/13_8.png)

- **MAML** :
  - Don't care about $\Theta$ performance in training task .
  - Care about the $\theta^{n}$ training by $\Theta$ .
- **Model Pre-training** :
  - Find the $\Theta$ that suit all task .
  - No guarantee that will get good $\theta^{n}$ training with $\Theta$ .

![13_9](/Li_DL/Notes/DL_Img/notes13/13_9.png)

Update Once, MAML can get good performance parameters.

Use the gradient of $\theta^{n}$ update $\Theta$.

![13_10](/Li_DL/Notes/DL_Img/notes13/13_10.png)

### Reptile

![13_11](/Li_DL/Notes/DL_Img/notes13/13_11.png)

**Different gradient update ways.**

![13_12](/Li_DL/Notes/DL_Img/notes13/13_12.png)

**MAML and Reptile : depend on gradient descent.**

## Gradient Descent as LSTM

![13_13](/Li_DL/Notes/DL_Img/notes13/13_13.png)

Input Gate and Forget Gate initialize artificially.

![13_14](/Li_DL/Notes/DL_Img/notes13/13_14.png)

Different from normal LSTM : $C_t$ will affect the $X_t$, the present parameters will affect next input.

**The initial parameters determine the direction**

![13_15](/Li_DL/Notes/DL_Img/notes13/13_15.png)

![13_16](/Li_DL/Notes/DL_Img/notes13/13_16.png)

Middle LSTM works like momentum, save the previous gradient descent.