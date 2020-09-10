# Basic Neural Networks

Basic Neural Networks transform an input through one or more hidden layers. Every layer is composed of a set of neurons, each of which is fully connected to all neurons in the prior layer. The final layer, also fully-connected to prior layer, is the output layer that provides the prediction.

## Binary Classification

* [Predict Sentiment of Tweets with Deep Neural Network](https://github.com/msfchen/deep_learning/tree/master/basicnn/tweetsentiment_dnn):
  - convert each tweet to a list of token_id
  - a batch data generator that provides equal number of positive and negative examples, optionally shuffled.
  - classifier using Trax framework; layers: Embedding -> Mean (average of word embeddings of a tweet) -> Dense -> LogSoftmax
  - Train: CrossEntropyLoss, Adam optimizer(0.01); Validation: CrossEntropyLoss, Accurary; Test Evaluation: Accuracy 99.31%