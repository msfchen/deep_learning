# Attention Models

- Given a set of vector values and a vector query, attention is a technique to compute a weighted sum of the values, dependent on the query.
- It is sometimes referred to as that the query attends to or focuses on particular parts of the values, by giving different weights to different parts.
- Attention variants by how attention scores are computed: basic dot-product attention, multiplicative attention, additive attention, scaled dot-product attention, content-based attention, location-based attention
- Attention variants by the query: when the query is a part of the set of the vector values themselves, we call it Self-Attention
- Attention variants by the span of the attention: Global vs Local Attention; Soft vs Hard Attention

## Neural Machine Translation

* [Neural Machine Translation with Seq2Seq Model with Multiplicative Attention](https://github.com/msfchen/deep_learning/tree/master/attentionmodel/translation):
  - The Spanish to English NMT system uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.
  - At each decoder timestep, the decoder hidden state is the query and all encoder hidden states are values. We get the attention scores using multiplicative attention.
  - We concatenate the attention output with the decoder hidden state and pass it through a linear layer, a Tanh, and a Dropout to attain the combined-output vector, which is used to produce a probability distribution over target words.
  - The loss at that timestep is the softmax cross entropy loss between the probability distribution and the actual target word. 
  - Beam Search Decoding is applied. We keep track of the k (beam size) most probable partial translations (hypotheses) on each step. For the highest-scoring hypothesis at end, we backtrack to obtain the full hypothesis.

