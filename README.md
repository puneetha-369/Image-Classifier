**Image Classifier Project**
This project implements an image classifier using PyTorch to identify flower species from images. The train.py script trains a neural network on a flower dataset and saves the model as a checkpoint. The predict.py script uses the saved model to predict the class of an input image, providing the top K predictions and their probabilities. The model supports custom architecture selection, hyperparameter tuning, and GPU acceleration for both training and inference.

**Project Overview**
The project is split into two main components:

1. Training a Model (train.py):
- Train a neural network on a dataset of flower images.
- Save the trained model as a checkpoint for later use.

2. Making Predictions (predict.py):
- Load the trained model from the checkpoint.
- Use the model to predict the class of a given image.
- Optionally map class indices to category names using a JSON file.
