Introduction
In the insurance industry, processing claims for vehicle damage is a common task.
With advancements in AI and Computer Vision, settling claims online by uploading damaged car images is now possible.

Dataset
https://www.kaggle.com/datasets/imnandini/analytics-vidya-ripik-ai-hackfest

Training set (train.zip)
Test set (test.zip)
Sample submission (sample_submission.csv)

Training Dataset
The training set contains a diverse dataset of car images with labels indicating the specific type of damage (e.g., dents, scratches, cracks).
The train.csv file includes the following columns:

image_id: Unique identifier of the image
filename: Filename of the image
label: Type of damage present in the car
Crack
Scratch
Tire Flat
Dent
Glass Shatter
Lamp Broken
Test Dataset
The test set contains only images, and the goal is to predict the type of damage for each image.
The test.csv file includes the following columns:

image_id: Unique identifier of the image
filename: Filename of the image
Sample Submission
The solution file must contain predictions for every image_id in the test set. It must contain only 2 columns - image_id and label.
The solution file format must be similar to that of sample_submission.csv. sample_submission.csv contains 2 variables:

image_id: Unique identifier of an image
label: Type of damage present in the car {1:crack, 2:scratch, 3:tire flat, 4:dent, 5:glass shatter, 6:lamp broken}
Evaluation Metric
The model will be evaluated based on the macro F1 score.

Table of Contents
Recognizing and Understanding Data

Reading train and test csv Files
Detailed EDA Check Images
Train Test Split

Build Dataset

Image Data Generator (Data augmentation)
flow_from_dataframe (with pandas dataframe)
flow_from_directory (with subdirectories)
Dataset Sanity Check
Dataset Benchmark Test
Optimize Performance
Explore Dataset Images
CNN (CONVOLUTIONAL NEURAL NETWORK) MODELING

Build CNN Model
Model Training
Model History
Save Model Weights
Model evaluation
BEST THRESHOLD calculate precision-recall curve
Predict Val Data
Predict Test Data
Save Submission
