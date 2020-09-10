# Word Vectors

Word vectors capture both syntactic and semantic features of words in a dense vector representation.

* [Train Word Vectors with CBOW Model](https://github.com/msfchen/deep_learning/tree/master/wordvector/cbow): 
  - In continuous bag of words (CBOW) model, we try to predict the center word given a few context words (the words around the center word).
  - A shallow neural network with one hidden layer is used; input is the average of all the one hot vectors of the context words; output is a softmax layer.
  - embs = (W1.T + W2)/2.0

* [Predict Word Relationships with Word Vectors](https://github.com/msfchen/deep_learning/tree/master/wordvector/analogies):
  - predict analogies between words using pre-trained word embeddings GoogleNews-vectors-negative300
  - Compare word embeddings by using a similarity measure (the cosine similarity).
  - Use PCA to reduce the dimensionality of the word embeddings and plot them in two dimensions.
  
* [Naive Machine Translation and Locality Sensitive Hashing](https://github.com/msfchen/deep_learning/tree/master/wordvector/translate_lsh):
  - Naive Word Translation
    - train a transformation matrix R that projects English embeddings X to French embeddings Y, by minimizing the the Frobenius norm ||X R -Y||^2
    - use 1-nearest neighbor algorithm to search for an embedding 𝐟 (as a row) in the matrix 𝐘 which is the closest to the transformed vector 𝐞𝐑
  - Find Most Similar Tweets
    - given a new tweet, find the top most similar ones from a tweet corpus 
    - a tweet is converted to a vector by the sum of all the word vectors of all the words it contains.
    - LSH provides an efficient way to find approximate K-NN