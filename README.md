# Dog vs Cat Classifier

## Introduction

The Dog vs Cat Classifier project aims to classify images as either dogs or cats using deep learning techniques. This project simplifies the classification task to distinguishing between two main categories: dogs and cats.

## Objective

The primary objective of this project is to develop a model capable of accurately differentiating between images of dogs and cats. The project involves tasks such as data preparation, model building, training, and evaluation.

## Dataset

The dataset used for this project consists of a large collection of images, each labeled as either a dog or a cat. The dataset is divided into training and testing sets, with images representing various breeds of dogs and different breeds of cats.

## Methodology

### Data Preparation

- Unzipping and organizing the dataset into separate folders for training and testing.
- Labeling images based on their file names to distinguish between dogs and cats.

### Model Building

- Constructing a convolutional neural network (CNN) model using the Keras Sequential API.
- Adding layers for convolution, pooling, batch normalization, dropout, and fully connected layers.
- Utilizing activation functions such as ReLU and softmax for feature extraction and classification.

### Model Training

- Training the model using the training data and validating its performance using the validation set.
- Applying data augmentation techniques such as rotation, shifting, zooming, and horizontal flipping to increase the diversity of the training data.
![Dog vs Cat](https://github.com/Toldblog/Dogs-Cats-Classifier/blob/main/example_image.png)
- Employing early stopping and learning rate reduction callbacks to prevent overfitting and improve convergence.

### Model Evaluation

- Evaluating the trained model's performance on both training and validation sets to assess accuracy and loss.
- Generating confusion matrix and classification report to analyze the model's performance across different classes (dogs and cats).

### Model Prediction

- Using the trained model to predict the labels (dog or cat) for the testing dataset.
- Visualizing the results by displaying sample images along with their predicted labels.

## Results

The model achieves a certain level of accuracy in distinguishing between images of dogs and cats, indicating its capability to generalize well to unseen data. Visualization of training and validation metrics helps in understanding the model's training progress and performance.

## Conclusion

The Dog vs Cat Classifier project demonstrates the application of deep learning techniques for binary image classification tasks. Further improvements can be made by experimenting with different architectures, hyperparameters, and data augmentation strategies to enhance the model's performance.

## Code and Documentation

For the full project code and documentation, please visit the [GitHub repository](https://github.com/Toldblog/Dogs-Cats-Classifier).
