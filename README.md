# Dimitraele-One-Class-Classification-Models-for-Anti-Doping-Research
Development of an LC-HRMS Workflow Using One-Class Classification Models for Detecting Unknowns in Doping Analysis


**Overview**
This script performs four classification methods using anomaly detection. The techniques used are 
a) One-Class Classification Support Vector Machine Model, 
b) Deep Learning Autoencoder,
c) One-Class k-Means, and
d) Principal Components Analysis.

Before running the classification code, ensure that you have set your working directory and loaded your dataset into R.

## **Prerequisites**
R version 3.6 or higher.
Necessary libraries: 
- keras: For deep learning and neural networks.
- ggplot2: For creating visualizations, including plots like the confusion matrix and scatter plots.
- e1071: Provides the implementation of SVM (Support Vector Machines), including the one-class classification used in the anomaly detection.
- caret: For building and evaluating machine learning models, particularly used here to create confusion matrices.
- ROSE: Used for oversampling techniques in imbalanced datasets.
- dplyr: For data manipulation, such as filtering, mutating, and summarizing data.
- tidyr: Used to tidy data, particularly reshaping the data frames.
- reshape2: For reshaping data, especially to convert data between wide and long formats, used in melt() for preparing data for ggplot.
- gridExtra: For arranging multiple plots in a grid, allowing for the creation of combined visualizations.

These libraries should be installed before running the script using install.packages().

## **Instructions**
### 1. Set Working Directory
Set the working directory to the folder where your dataset is located. Replace "path/to/your/directory" with the actual path:
setwd("path/to/your/directory")

### 2. Import the Dataset
Ensure that your dataset is saved as a CSV file. Replace "data.csv" with the actual filename of your dataset:
data <- read.csv("data.csv", header = TRUE)

### 3. Run the Classification Script
After setting the working directory and importing the data, proceed with the provided classification code.

## **Example in R**
setwd("C:/Users/YourUsername/Documents") # Set working directory

Import the dataset
data <- read.csv("my_data.csv", header = TRUE) # Run classification code (provided separately)

## **Notes**
Ensure the dataset is properly formatted, with the necessary columns matching the expected structure in the classification code.
