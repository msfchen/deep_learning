# Basic Neural Networks

Basic Neural Networks transform an input through one or more hidden layers. Every layer is composed of a set of neurons, each of which is fully connected to all neurons in the prior layer. The final layer, also fully-connected to prior layer, is the output layer that provides the prediction.

## Multiclass Classification

* [Neural Dependency Parsing](https://github.com/msfchen/deep_learning/tree/master/basicnn/dependencyparser):
  - A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between head words, and words which modify those heads.
  - In a transition-based parser, at every step, the parser applies one of the three transitions: SHIFT, LEFT-ARC, and RIGHT-ARC.
  - We will train a neural netwrok to predict which transition should be applied next, with the goal of maximizing performance on UAS (Unlabeled Attachemnt Score) metric.
  - PyTorch; the feature vector consists of a list of tokens (e.g., the last word in the stack, first word in the buffer, etc.) that is represented as a list of integers, which is then converted into a single concatenated embedding. The training is to minimize cross-entropy loss. 

## Binary Classification

* [Predict Sentiment of Tweets](https://github.com/msfchen/deep_learning/tree/master/basicnn/tweetsentiment_dnn):
  - convert each tweet to a list of token_id
  - a batch data generator that provides equal number of positive and negative examples, optionally shuffled.
  - classifier using Trax framework; layers: Embedding -> Mean (average of word embeddings of a tweet) -> Dense -> LogSoftmax
  - Train: CrossEntropyLoss, Adam optimizer(0.01); Validation: CrossEntropyLoss, Accurary; Test Evaluation: Accuracy 99.31%
  
## Regression

* [Predict Bike Rental Count](https://github.com/msfchen/deep_learning/tree/master/basicnn/bikerental):
  - structured data requires data exploration, feature extraction, and feature engineering. The target value is hourly rental count. 
  - time series prediction data split: the last 21 days for test, the further prior 60 days for validation, all earlier 643 days for training, out of 2-year historical data
  - the simplest form of NN is used: one input layer, one hidden layer, and one output neuron; MSE is used as loss;