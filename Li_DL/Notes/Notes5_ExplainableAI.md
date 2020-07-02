# Explainable AI

![5_1](DL_Img/notes5/5_1.png)

**Not only get good accuracy, but also know why.**

![5_2](DL_Img/notes5/5_2.png)

![5_3](DL_Img/notes5/5_3.png)

## Local Explanation

**Explain the Decision** : Why do you think this image is a cat?

![5_4](DL_Img/notes5/5_4.png)

### Gradient Based Approaches

![5_9](DL_Img/notes5/5_9.png)

**Bigger gradient, most important component.**

![5_5](DL_Img/notes5/5_5.png)

### Attack Interpretation

![5_6](DL_Img/notes5/5_6.png)

**By use samll noises to find the most important component.**

## Global Explaination

**Explain the Whole Model** : What do you think a 'cat' looks like?

![5_7](DL_Img/notes5/5_7.png)

![5_10](DL_Img/notes5/5_10.png)

### Constraint from Generator

![5_8](DL_Img/notes5/5_8.png)

## Using a model to explain another

**Some models are easier to interpret. Using interpretable model to minic uninterpretable models.**

![5_11](DL_Img/notes5/5_11.png)

- **Agnostic Explanations (LIME)**
  - 1. Given a data point you want to explain.
  - 2. Sample at the nearby.
  - 3. Fit with linear model(or other interpretable models).
  - 4. Interpret the linear model.

### Decision Tree

![5_12](DL_Img/notes5/5_13.png)

![5_13](DL_Img/notes5/5_12.png)