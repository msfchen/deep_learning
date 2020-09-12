# Generative Adversarial Networks

- Generative Adversarial Networks (GANs) 

## 

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
    - load pre-trained VGG-19 model and freeze "features" layers parameters; change the output layer size from 1000 to 133 (the number of dog breeds in our training data)
    - CrossEntropyLoss; Adam(0.001) Optimizer for "classifier" layers parameters; test evaluation: accuracy


