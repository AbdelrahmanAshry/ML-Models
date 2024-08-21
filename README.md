# Simplified DenseNet for Multi-Class Image Classification

## Overview

This project implements a simplified version of the DenseNet architecture for a multi-class image classification task. The model is designed to classify images into one of seven categories using a dataset with the following sizes:

- Training: 3087 images
- Validation: 1028 images
- Test: 1508 images

## DenseNet Architecture

### DenseNet Overview

DenseNet is a type of convolutional neural network where each layer is connected to every other layer in a feed-forward fashion. For each layer, the feature maps of all preceding layers are used as inputs, and its own feature maps are used as inputs into all subsequent layers. This connectivity pattern introduces several benefits:

1. **Alleviates the Vanishing Gradient Problem**: Direct connections between layers help gradients flow directly through the network, improving training in very deep networks.
2. **Feature Reuse**: DenseNet allows feature reuse across layers, reducing the need for a very large number of parameters.
3. **Parameter Efficiency**: Despite being deep, DenseNet is parameter-efficient, reducing overfitting in scenarios with smaller datasets.

### Simplified DenseNet Model

In this project, we implement a simplified version of DenseNet, which consists of the following components:

- **Convolutional Layer**: The model starts with a standard convolutional layer to process the input image, followed by batch normalization, ReLU activation, and max-pooling.
  
- **Dense Blocks**: Three dense blocks are implemented, each containing several convolutional layers. The number of feature maps increases by a fixed "growth rate" after each layer.
  
- **Transition Layers**: After each dense block, a transition layer reduces the number of feature maps and performs downsampling.
  
- **Fully Connected Layer**: After the final dense block, the feature maps are pooled and passed through a fully connected layer that outputs the class probabilities.


### Data Pre Processing
The images are first resized to 224x224 pixels, which is a typical input size for CNNs. The preprocessing pipeline includes the following steps:

1. **Random Horizontal Flip**: Helps the model generalize better by flipping the images horizontally during training.
2. **Random Rotation**: Introduces additional variability by randomly rotating the images by up to 10 degrees.
3. **Normalization**: The images are normalized using mean and standard deviation values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively, which are common values for pre-trained models on ImageNet.

### Model Architecture

## Simple Dense model 

  Initial Convolution
  
  DenseBlock

TransitionLayer

 DenseBlock 2

 TransitionLayer 2
    
 DenseBlock 3
 TransitionLayer 3

  Avgpool Layer 
 Fully connected Layer


### Model Training
### Training Parametars

**Optmizer**: We use the Adam optimizer with a learning rate of 0.001. Adam is chosen for its adaptive learning rate capabilities, making it well-suited for a model that has to generalize well on a small dataset.

**Loss Function**: CrossEntropyLoss is used as it is standard for multi-class classification problems.

**Epochs**: The model is trained for 10 epochs. This is a starting point; more epochs can be used if the validation loss continues to decrease.

### Tuning Parametars

**Growth Rate**: The growth rate of the DenseNet can be adjusted depending on the complexity of the dataset. Increasing it will increase the number of feature maps, which may improve performance at the cost of additional computational overhead.

**Number of Layers** in Dense Block: Adding more layers in each dense block will allow the model to capture more complex features but might lead to overfitting on small datasets.

**Learning Rate**: The learning rate could be tuned using techniques like learning rate scheduling or grid search.

### Future Work

**Deeper Dense NET**: Experiment with a deeper DenseNet architecture to potentially increase accuracy.

**Ensemble methods**: Combine predictions from multiple models to improve overall accuracy.

**Data Augmentation**: Explore more aggressive data augmentation strategies to increase the effective size of the training set.

### Conclusion
The simplified DenseNet architecture implemented in this project is designed to provide a good balance between model complexity and performance for small datasets. The model and training pipeline offer a strong foundation that can be further improved with tuning and additional data augmentation strategies. This architecture should serve as a solid baseline for image classification tasks with limited data.
