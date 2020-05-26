# # Notes.3 RNN

## RNN

### RNN

***Change the sequence order will change output.***

***RNN-Based Network is not always easy to learn.***

由于多个 step 共享参数 W,训练过程中 RNN 的 Gradient 可能会巨大或者巨小，因此在训练是需要 **Clipping** 限制 Gradient 的步长，即使用较小的学习率。 

#### Type

| **Type** | **Input** | **Output**| **Application** |
| --------- | -------- | --------- | -------- |
| **Many to One** | Sequence | Vector | Classification |
| **Many to Many** | Sequence | Sequence(much shorter than input) | Speech Recognition (CTC:Conectionist Tenpiral Classification)|
| **Many to Many** | Sequence | Sequence(similar with input) | Machine Translation |
| **Beyond Sequence** | Syntacitc Parsing | Sequence | Sentence Analysis |

### LSTM

#### 

1. **LSTM can deal with gradient vanish (but not gradient explore).**

2. **Learning rate should be small.**

3. **Memory and input are added.(RNN Memory will be cleaned)**

4. **The influence never disappear unless forget gate is closed.**

### GRU

Simpler than LSTM

### RNN v.s. Structured Learning

| RNN LSTM | HMM.CRF.Structure Perception.SVM|
| --- | --- |
| Unidirectional RNN does not consider the whole sequence | Using Viterbi,so consider the whole sequence |
| Cost and error not always related | Can explicitly consider the label dependency |
| Deep | Cost is the upper bound of error |

**Deep is better.**

***MLDS:Machine Learning and having it deep and structures***


