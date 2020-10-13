# Attention Models

- Given a set of vectors of keys, values, and query, attention is a technique to compute a weighted sum of the values, dependent on the relevance scores between the query and keys.
- It is sometimes referred to as that the query attends to or focuses on particular parts of the keys or values, by giving different weights to different parts. 
- Attention Variants:

    | By | Variants |
    |-----------------------------------|--------------------------------------|
    | how attention scores are computed | basic dot-product attention, multiplicative attention, additive attention, scaled dot-product attention |
    | query-key relative location | Encoder-Decoder Attention, Causal Self Attention, Bi-directional Self Attention, Causal with Prefix Self Attention |
    | span of the attention | Global vs Local Attention; Soft vs Hard Attention; Locality Sensitive Hashing Attention |

## Encoder-Decoder Attention Models

* [Neural Machine Translation with Seq2Seq Model with Multiplicative Attention](https://github.com/msfchen/deep_learning/tree/master/attentionmodel/translation):
  - The Spanish to English NMT system uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.
  - At each decoder timestep, the decoder hidden state is the query and all encoder hidden states are values. We get the attention scores using multiplicative attention.
  - We concatenate the attention output with the decoder hidden state and pass it through a linear layer, a Tanh, and a Dropout to attain the combined-output vector, which is used to produce a probability distribution over target words.
  - The loss at that timestep is the softmax cross entropy loss between the probability distribution and the actual target word. 
  - Beam Search Decoding is applied. We keep track of the k (beam size) most probable partial translations (hypotheses) on each step. For the highest-scoring hypothesis at end, we backtrack to obtain the full hypothesis.

## Transformer Language Models 

* [Article Summarization with Transformer Decoder](https://github.com/msfchen/deep_learning/tree/master/attentionmodel/transf_summarizer):
  - Training data are in the format of [Article][\<EOS\>][\<pad\>][Summary][\<EOS\>].
  - Training examples are batched by grouping similar lengthed examples in buckets, with varying bucket sizes depending on example length.
  - Transformer Decoder; Masked Multi-Head Self-Attention; Causal Attention
  - At inference time, an article is fed into the model and the model generates a summary one word at a time, using greedy decoding algorithm, until \<EOS\> token is generated.

* [Question Answering with T5 Model](https://github.com/msfchen/deep_learning/tree/master/attentionmodel/t5_qa):
  - Subword Tokenization
  - BERT

* [Chatbot with Reformer Model](https://github.com/msfchen/deep_learning/tree/master/attentionmodel/reformer_chatbot):
  - Training data are in the format of [Person 1:][message 1][Person 2:][message 2]...[Person 1:][message N][Person 2:][message N], where N is around 5.
  - Training examples are batched by grouping similar lengthed examples in buckets, with varying bucket sizes depending on example length.
  - Build Reformer Language Model; LSH Attention reduces computation cost; Reversible Layers reduce momory cost.
  - At inference time, [Person 1:][message 1][Person 2:] is fed into the model and the model generates subsequent dialogues one word at a time until the given max_len of tokens have been generated.