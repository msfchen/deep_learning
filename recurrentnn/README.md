# Recurrent Neural Networks

- Recurrent neural networks are designed to model sequence, in which the hidden state of the previous step is an input to the current step. The same set of parameter values are applied repeatedly to every steps of the sequence.
- Long sequences tend to have vanishing gradient problem that give rise to LSTM and GRU units as well as other fixes, such as gradient clipping and skip connections.
- More fancy RNN variants: Bidirectional RNNs and Multi-layer RNNs. 

## Language Model

* [Predict Next Character with Recurrent Neural Network](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/predictnextchar):
  - convert each sentence to a list of character token_ids, ending with EOS_int.
  - a batch data generator, optionally shuffled.
  - Gated Recurrent Unit (GRU) model using Trax framework; layers: ShiftRight -> Embedding -> n_layers of GRU -> Dense -> LogSoftmax
  - Train: CrossEntropyLoss, Adam optimizer(0.0005); Validation: CrossEntropyLoss, Accuracy; Test Evaluation: Perplexity
  - generating sentence, one predicted next character at a time
* [Novel Writing with Character-Level RNN](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/textgenbychar)
* [TV-script Generation with Word-Level RNN](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/tvscriptgeneration)

## Word Tagging

* [Named Entity Recognition](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/ner):
  - explore the pre-processed labelled data (B-, I-, O)
  - a batch data generator, optionally shuffled.
  - Long Short-Term Memory (LSTM) model using Trax framework; layers: Embedding -> LSTM -> Dense -> LogSoftmax
  - Train: CrossEntropyLoss, Adam optimizer(0.01); Validation: CrossEntropyLoss, Accuracy; Test Evaluation: Accuracy (95.4%)

## Neural Machine Translation

* [Character-based Neural Machine Translation](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/characternml):
  - The Spanish to English NMT system uses a character-based 1-D convolutional encoder and a word-level LSTM decoder plus a character-level LSTM decoder that will kick in when the word-level decoder produces an \<UNK\> token. Character-level decoder generates the target word one character at a time, which can produce rare and out-of-vocabulary target words.
  - Encoder Architecture: convert word to char idxs -> padding and embedding lookup -> MaxPool(ReLU(1-D Conv)) -> Highway Network Layer (with skip-connections) and Dropout
  - Character-level Decoder Architecture: char idxs -> char embeddings -> unidirectional LSTM -> linear layer -> softmax -> sum of char-level cross-entropy loss
  - Greedy decoding algorithm (as opposed to beam search algorithm) is used to generate the sequence of characters.

## Siamese Networks

A Siamese Network, also known as Twin Network, is composed of two identical networks that share the same weights while working in parallel on two different input vectors to compute similarity measures of of the corresponding output vectors.

* [Predict Duplicate Questions](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/predictdupquests):
  - explore the pre-processed is_duplicate labelled question pairs
  - Only use duplicate question pairs to prepare training data so that data generator will produce batches ([q1_1, q1_2, q1_3,...], [q2_1, q2_2, q2_3, ...]) where q1_i and q2_k are duplicate if and only if i = k.
  - tokenize each question => build vocab {token : idx} => convert questions to tensors; split train/valid to 8:2.
  - a batch data generator, optionally shuffled, that returns two lists of vectors of shape (batch_size * max_len)
  - Siamese Network using Trax framework; layers: Embedding -> LSTM -> Mean (average word vectors of each question output) -> Normalize (because cosine similarity = dot product of normalized vectors)
  - Triplet Loss Function with Hard Negative: A (anchor), P (positive), N (negative); Loss(A, P, N) = mean(Loss1 + Loss2); Loss1 = max(-cos(A, P) + mean_neg + alpha, 0); Loss2 = max(-cos(A, P) + closest_neg + alpha, 0)
  - Train: TripletLoss, Adam optimizer(0.01), lr_schedule = trax.lr.warmup_and_rsqrt_decay(400, 0.01); Validation: TripletLoss; Test Evaluation: Accuracy (69.1%)

## Time Series

* [Simple Time Series Prediction](https://github.com/msfchen/deep_learning/tree/master/recurrentnn/timeseries):
  - given a time series [n1, n2, ..., nt]; use input [n1, n2, ..., nt-1] and output [n2, ..., nt] to train a RNN so that it can predict the next item in a given test series.