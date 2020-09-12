# Convolutional Neural Networks

- Convolutional neural networks (CNNs) make the explicit assumption that the inputs are images with 3 dimensions: width, height, depth. Therefore, the layers of a CNN have neurons arranged in 3 dimensions.
- There are three main types of layers to build CNN architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). 
- A convolutional layer contains a set of filters with learnable parameters. The height and width (receptive field) of the filters are smaller than those of the input volume. Filters are slid (convolved) across the width and height of the input and the dot products between the input and filters are computed at every spatial position. The output volume of the convolutional layer is obtained by stacking the activation maps of all filters along the depth dimension. 
- A pooling layer will perform a downsampling operation along the spatial dimensions (width, height).

## Image Classification

* [Dog Breed Identification](https://github.com/msfchen/deep_learning/tree/master/convolutionalnn/dogbreed):
  - Goal: for a given dog image, provide an estimate of the dog's breed; for a given human image, provide the most resembling dog breed.  
  - using pre-trained models:
    - use a pre-trained Open Source Computer Vision Haar Cascades classifier to detect human face.
    - use a pre-trained deep CNN, VGG-16, model to predict dogs. VGG-16 is trained to classify 1000 categories of objects, of which category idx 151 ~ 268 (inclusive) are dogs.
  - buld CNN model from scratch:
    - augment image data by transformations, such as resize, center crop, random horizontal flip, and random rotation; data splitted into train, valid, test sets.
    - model architecture: 4 times of Pool(ReLU(Conv)) + flatten + 2 times of Dropout(ReLU(Linear)) + Linear
    - CrossEntropyLoss; Adam(0.0007) Optimizer; test evaluation: accuracy
  - build CNN model by transfer learning from a pre-trained model:
    - start from pre-trained VGG-19 model; change the output layer size from 1000 to 133 (the number of dog breeds in our training data)
    - CrossEntropyLoss; Adam(0.001) Optimizer; test evaluation: accuracy

## Image Style Transfer

* [Combine the content of one image with the style of another image](https://github.com/msfchen/deep_learning/tree/master/convolutionalnn/styletransfer):
  - analyses of outputs of each layer of deep CNNs indicate that earlier layers capture lower level features, such as directional edges, colors, and color edges; and later layers capture more complex shapes, such as mouth, eyes, etc. 
  - load the pre-trained VGG-19 model and freeze its parameters; load and normalize the two images; extract features at each convolutional layer from passed-in images; initialize a target image copied from content image.
  - run a learning process with Adam optimizer to update the target image to minimize the total loss:
    - total_loss = content_weight * content_loss + style_weight * style_loss
    - content_loss = mean((target_features['conv4_2'] - content_features['conv4_2'])**2) where conv4_2 is the 2nd from the last Conv layer
    - style_loss = sum over all layers of layer_style_loss / (d * h * w) where layer_style_loss = style_weights[layer] * mean((target_gram - style_gram)**2)

## Image Compression

* [Convolutional Autoencoder](https://github.com/msfchen/deep_learning/tree/master/convolutionalnn/autoencoder):
  - A compressed representation of images can save storage space and enable more efficient sharing.
  - The encoder portion will be made of convolutional and pooling layers and the decoder will be made of transpose convolutional layers that learn to reconstruct a compressed representation.