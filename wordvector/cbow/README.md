# Train Word Vectors with CBOW Model

- The Continuous bag of words model
  - In continuous bag of words (CBOW) model, we try to predict the center word given a few context words (the words around the center word).
  - A shallow neural network with one hidden layer h is used; input X is the average of all the one hot vectors of the context words; output is a softmax layer.
  - h = W1X + b1; a = ReLU(h); z = W2a + b2; y = softmax(z)

- Training the Model
  - V = size of vocab; N = dim of hidden layer, and the eventual word vector.
  - def initialize_model(N,V, random_seed=1) returns randomly initialized W1 of (N, V), W2 of (V, N), b1 of (N, 1), b2 of (V, 1)
  - def forward_prop(x, W1, W2, b1, b2) returns z and h, with shape of (V, 1) and (N, 1), respectively.
  - cross-entropy cost function: 
    - logprobs = np.multiply(np.log(yhat),y) + np.multiply(np.log(1 - yhat), 1 - y)
    - cost = - 1/batch_size * np.sum(logprobs)
    - cost = np.squeeze(cost)
  - def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size) returns gradients of matrices and biases, grad_W1, grad_W2, grad_b1, grad_b2
    - Compute l1 as W2^T (Yhat - Y)
  - def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03)
    - W1, W2, b1, b2 = initialize_model(N,V, random_seed=282)
    - mini-batch size = 128
    - in each batch:
      - z, h = forward_prop(x, W1, W2, b1, b2)
      - yhat = softmax(z)
      - cost = compute_cost(y, yhat, batch_size)
      - grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
      - Update weights and biases by - alpha * grad
    - return W1, W2, b1, b2

- word vectors are defined as (W1.T + W2)/2.0
