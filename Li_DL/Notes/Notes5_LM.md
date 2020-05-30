# Language Model

## Word Embedding

## Transformer

![transformer](DL_Img/notes5/transformer.png)

### Encoder-Decoder

#### Encoder

The stack of 6 identical layers, each layer has two sublayers, first is ***muilti-head self attention mechanism***, second is ***position-wise fully connected feed-forward network***.

#### Decoder

The stack of 6 identical layers, each layer has three sublayers, first is ***multi-head self attention with subesequence mask*** ,second is ***multi-head self attention***, third is ***position-wise fully connected feed-forward network***.

Employing a ***residual connection*** around each of the two sub-layers, folloed by layer normlization. $LayerNorm(x + Sublayer(x))$.

### Attention

![transformer-attn](DL_Img/notes5/attention.png)

#### Scaled Dot-Product Attention

$$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_K}})V$$

In Encoder's first sublayer and Decoder's second sublayer, **Q,K,V all come from Input embedding**.

In Decoder's second sublayer, **Q from Decoder first sublayer output; K,V from Encoder output**.

#### Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$MultiHead(Q, K, V) = Concat(head1, ..., head_h)W^O\\
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V,)$$

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

![MLM](DL_Img/notes5/MLM.png)

To mitigate the downside that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. 

#### Next Sequence Prediction

Specifically, when choosing the sentences A and B for each pre- training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). 

![NSP](DL_Img/notes5/NSP.png)

### Fine-tuning

BERT use self-attention machanism to catch the bidirectional cross attention between two sequence.

![bert-fine-tuning](DL_Img/notes5/bert-fine-tuning.png)

- the sentence A and B from pretraining:
    - Sentence pairs in paraphrasing
    - Hypothesis-premise pairs in entailment
    - Question-passage pairs in question answering
    - A degenerate text-∅ pair in text classification or sequence tagging

## GPT

## ELMO

![bert-elmo-gpt](DL_Img/notes5/bert-gpt-elmo.png)

## XLNet

## Reformer