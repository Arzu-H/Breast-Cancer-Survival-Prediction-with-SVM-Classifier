# Breast Cancer Survival Prediction

This project aims to predict whether a breast cancer patient will be **Alive** or **Deceased** based on clinical and molecular data.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Evaluation](#evaluation)
- [Additional Steps and Suggestions](#additional-steps-and-suggestions)
- [License](#license)

## Introduction
The goal of this project is to analyze a dataset of breast cancer patients and build a classification model to predict their survival status. The approach includes:
- Data exploration and visualization
- Preprocessing and feature selection
- Training a **Support Vector Machine (SVM)** classifier
- Evaluating the model’s performance

## Dataset
The dataset (`BRCA.csv`) contains clinical and molecular information about breast cancer patients. Key features include:
- **Demographic Information**: `Patient_ID`, `Age`, `Gender`
- **Biomarkers**: `Protein1`, `Protein2`, `Protein3`, `Protein4`
- **Cancer Stage & Histology**: `Tumour_Stage`, `Histology`
- **Receptor Status**: `ER_status`, `PR_status`, `HER2_status`
- **Treatment & Follow-up**: `Surgery_type`, `Date_of_Surgery`, `Date_of_Last_Visit`
- **Target Variable**: `Patient_Status` (Alive/Dead)

## Installation
To run this project, install the required Python dependencies:
```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

## Usage
1. Load the dataset and explore the distributions of key features.
2. Preprocess the data (handle missing values, categorical encoding, scaling).
3. Train an **SVM classifier** to predict patient survival status.
4. Evaluate model performance using accuracy and other metrics.

Run the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

## Model Overview
The pipeline follows these steps:
- **Feature Engineering**: Encoding categorical variables and scaling numerical features.
- **Model Selection**: Using **Support Vector Machines (SVM)** for classification.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score.

## Evaluation
The model is evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

## Additional Steps and Suggestions
- **Try Different Models**: Experiment with Logistic Regression, Random Forest, or Deep Learning models.
- **Feature Selection**: Use techniques like PCA or feature importance scores.
- **Hyperparameter Tuning**: Optimize SVM parameters using GridSearchCV.
- **Deploy the Model**: Convert the trained model into a web API using Flask or FastAPI.

## License
This project is for educational and research purposes only.


