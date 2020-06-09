# Language Model

## Transformer

Transformer : seq2seq model with **'self-attention'**.All of the ouput can be compute parallelly.

RNN hard to parallel.

CNN can parallel but small step,only higher layer can consider longer sequence.

![transformer](DL_Img/notes5/transformer.png)

### Encoder-Decoder

#### Encoder

The stack of 6 identical layers, each layer has two sublayers, first is ***muilti-head self attention mechanism***, second is ***position-wise fully connected feed-forward network***.

#### Decoder

The stack of 6 identical layers, each layer has three sublayers, first is ***multi-head self attention with subesequence mask*** ,second is ***multi-head self attention***, third is ***position-wise fully connected feed-forward network***.

Employing a ***residual connection*** around each of the two sub-layers, folloed by layer normlization. $LayerNorm(x + Sublayer(x))$.

**Layer Norm & Batch Norm** 

Layer norm always use with RNN,regardless of the batch.

### Attention

**Dynamic Conditional Generation**

![transformer-attn](DL_Img/notes5/attention.png)

#### Scaled Dot-Product Attention

$$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_K}})V$$

In Encoder's first sublayer and Decoder's second sublayer, **Q,K,V all come from Input embedding**.

In Decoder's second sublayer, **Q from Decoder first sublayer output; K,V from Encoder output**.

#### Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$MultiHead(Q, K, V) = Concat(head1, ..., head_h)W^O\\
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V,)$$

### Position embedding

input is $x_i$ and embedding is $W^I x_i = \alpha_i$,and position embedding is $W^P p_i = e_i$,the input embedding is $\alpha_i + e_i$.Two embedding is added but not concat.

$[ W^I | W_P ] * concat(x_i, p_i) = concat(W^I, W^P) = \alpha_i + e_i$

### Position-wise Feed-forward networks

$$FFN(x) = max(0,xW_1+b_1)W_2+b_2$$

The dimensionality of input and output is $d_{model} = 512$，and the inner-layer has dimensionality $d_{ff} = 2048$.*

### Mask

#### Padding mask

Use 0 to complete the input sequence to the fixed length.

#### Sequence mask

![mask](DL_Img/notes5/transformer-mask.png)

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

## BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers. 

Learned from a large amount of test without annotation.

The bert use character feature not word character to learn the representation of sentence.Such as '葡萄' while be split to '葡' and '萄' to train the model.
### Input Representations

![bert_embedding](DL_Img/notes5/bert-embed.png)

**Tokens embedding** :

Use WordPiece embeddings with a 30000 token vocabulary.

**Segment embedding** :

Use a learned embedding to every token indicating whether it belongs to sentence A or sentence B.

**Position embedding** :

Use learned position embedding to represent the position information.

### Pre-training

#### Masked LM

In order to train a deep bidirectional representation, simply mask some percentage of the input tokens at random, and then predict those masked tokens. In contrast to denoising auto-encoders, BERT only predict the masked words rather than recon-structing the entire input.

- 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
- 10% of the time: Replace the word with a random word, e.g., my dog is hairy → my dog is apple
- 10% of the time: Keep the word unchanged,e.g., my dog is hairy → my dog is hairy.

The purpose of this is to bias the representation towards the actual observed word.

To mitigate the downside that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. 

**Predict the masked word:** Setence with [MASK] → BERT model → Linear Multi-Class classifier($d_{vocab}$)

#### Next Sequence Prediction

Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). 

- Input: [CLS] the man went to [MASK] stores [SEP] he bought a gallon [MASK] milk [SEP]
  Output: IsNext
- Input: [CLS] the man went to [MASK] stores [SEP] penguin [MASK] are flight ##less birds [SEP]
  Output: NotNext

**[CLS]** : Classification results
**[SEP]** : Boundary of two sentences

**Predict the sentence pair:** Sentence pair → BERT model → [CLS] → Linear-Binary Classifier → 0 or 1

**Using simple classifier in MLM and NSP, ensure that the model has a strong learning ability.**

**Both part used at the same time.**

### Fine-tuning

BERT use self-attention machanism to catch the bidirectional cross attention between two sequence.

![bert-fine-tuning](DL_Img/notes5/bert-fine-tuning.png)

- The sentence A and B from pretraining:
    - Sentence pairs in paraphrasing
    - Hypothesis-premise pairs in entailment
    - Question-passage pairs in question answering
    - A degenerate text-∅ pair in text classification or sequence tagging

- **Downstream tasks**
  - **Sentiment / Document Classification** 
  Fine-tune BERT → [CLS] → Linear classifier(Trained from scratch) → Class
  - **Tagging / Slot filling** 
  Character → Fine-tune BERT → Linear Classifier → Class
  - **Nature Language Inference** 
  Sentence pair → Fine-tune BERT → Linear Classifier → Class [T/F/UNK]
  - **Extraction-Based QA** 
  Document(s) and Question(e) (s and e learned from scratch)→ Fine-tune BERT → Document representaion {$d_0, ... d_n$} → s and e dot product with $d_i$ → Answer {$d_s, ... d_e$} if s != e else : No answer

## GPT

Generative Pre-Training : Transformer Decoder : self-attention generate model.

## ELMO

Embedding from Language Model

Bidirectional-RNN-Based LM : Predict next token

Embedding : $ h = \alpha_1 h_1 + \alpha_2 h_2$

$\alpha_1, \alpha_2$ learned with the down stream tasks.

![bert-elmo-gpt](DL_Img/notes5/bert-gpt-elmo.png)

## XLNet

## Reformer