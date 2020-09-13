# Generative Adversarial Networks

- A Generative Adversarial Network (GAN) is composed of two adversarial networks, a discriminator and a generator. 
- The discriminator is a classifier that is trained to classify real or fake data. 
- The generator generates fake data from a random vector (a latent vector in a latent space). As the generator trains, it learns how to map latent vectors to recognizable data that can fool the discriminator.

## Image Generation

* [Hand-written Digits Generation with Deep GAN](https://github.com/msfchen/deep_learning/tree/master/gan/digitsimple):
  - Discriminator: 3 times of dropout(leaky_relu(Linear)) -> Linear  
  - Generator: 3 times of dropout(leaky_relu(Linear)) -> tanh(Linear)
  - The losses will be binary cross entropy loss with logits, BCEWithLogitsLoss, which combines a sigmoid activation function and binary cross entropy loss in one function. total_loss = real_loss + fake_loss. For generator loss, the labels are flipped.
  - Training will involve alternating between training the discriminator optimizer and the generator optimizer.
  
* [Street View House Numbers Generation with Deep Convolutional GAN](https://github.com/msfchen/deep_learning/tree/master/gan/housenumconv):
  - Discriminator: leaky_relu(conv) -> 2 times leaky_relu(BatchNorm(conv) -> flatten -> Linear
  - Generator: Linear -> de-flatten -> 2 times relu(BatchNore(transpose conv)) -> tanh(transpose conv)
  - The losses will be binary cross entropy loss with logits, BCEWithLogitsLoss, which combines a sigmoid activation function and binary cross entropy loss in one function. total_loss = real_loss + fake_loss. For generator loss, the labels are flipped.
  - Training will involve alternating between training the discriminator optimizer and the generator optimizer.

* [Face Image Generation with Deep Convolutional GAN](https://github.com/msfchen/deep_learning/tree/master/gan/facegen):
  - Discriminator: leaky_relu(conv) -> 2 times leaky_relu(BatchNorm(conv) -> flatten -> Linear
  - Generator: Linear -> de-flatten -> 2 times relu(BatchNore(transpose conv)) -> tanh(transpose conv)
  - The losses will be binary cross entropy loss with logits, BCEWithLogitsLoss, which combines a sigmoid activation function and binary cross entropy loss in one function. total_loss = real_loss + fake_loss. For generator loss, the labels are flipped.
  - Training will involve alternating between training the discriminator optimizer and the generator optimizer.
  
## Image-to-Image Translation

- Given two sets (domains) of unordered and unpaired images, learn to transform images from one domain to another. This is an unsupervised learning, because these images do not come with labels. Also, there is no exact correspondences between individual images in those two sets.
- Examples of domains: summer vs winter, Monet painting vs landscape photos, zebras vs horses, areial photos vs street map 

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/msfchen/deep_learning/tree/master/gan/cyclegan):
  - Goal: translate images from summer scene to winter scene or vice versa
  - Main Idea: two Discriminators, one for each domain; two CycleGenerators, one for each translation; Cycle-consistency loss: x → G(x) → F(G(x)) ≈ x and  y → F(y) → G(F(y)) ≈ y
  - Residual Function = the difference between a mapping applied to x and the original input x; In our case, Cycle-consistency loss is a residual function.
  - Discriminator: ReLU(Conv) -> 3 times of ReLU(BatchNorm(Conv)) -> Conv 
  - CycleGenerator: 3 times of ReLU(BatchNorm(Conv)) -> n times of (x + BatchNorm(Conv(ReLu(BatchNorm(Conv(x)))))) -> 2 times of ReLu(BatchNorm(ConvTranspose)) -> tanh(ConvTranspose)
  - real_MSE_loss = mean((D_out - 1)\*\*2); fake_MSE_loss = mean(D_out\*\*2); cycle_consistency_loss = lambda_weight\*mean(abs(real_im - reconstructed_im))
  - Alternating between training the discriminators and the generators, for a specified number of training iterations. 