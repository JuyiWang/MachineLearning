# Adversarial Attack

## Attack

Find the most effective $\Delta x$ to fool the  Network.

![6_1](DL_Img/notes6/6_1.png)


### How to attack?

![6_2](DL_Img/notes6/6_2.png)

**Non-Target Attack** : Stay far from $y_{true}$

**Target Attack** : Far from the correct answer, but close to the specific error

![6_3](DL_Img/notes6/6_3.png)

**L-infinity applies to Image Attack** : Change every pixel a little bit, not change one pixel much.

![6_4](DL_Img/notes6/6_4.png)

Replace $x$ with the $x_t$ that satisfies the condition and is closest to the original $x$.

![6_5](DL_Img/notes6/6_5.png)

- **Different Attack Method** :
  - **Different optimization methods**.
  - **Different constraints**.

### FGSM

![6_6](DL_Img/notes6/6_6.png)

**Consider only the direction, not the magnitude.**

Is equivalent to using a large Learning Rate.

### White Box v.s. Black Box

![6_7](DL_Img/notes6/6_8.png)

![6_8](DL_Img/notes6/6_9.png)

### Unversal Adversarial Attack

Find an special $\Delta x$ that make all inputs wrong .

![6_10](DL_Img/notes6/6_10.png)

## Defense

![6_11](DL_Img/notes6/6_11.png)

### Passice Defense

**Add a shield : Finding the attacked Image without modifying the model.**

![6_12](DL_Img/notes6/6_12.png)

### Proactive Defense

**Training a model that is robust to adversarial attack.**

![6_13](DL_Img/notes6/6_13.png)

**Each iteration will find new vulnerabilities, but defense algorithm A can only work against attack algorithm A, not for attack algorithm B.**