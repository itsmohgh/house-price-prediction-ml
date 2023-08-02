# house-price-prediction-ml
This repo contains Python code for house price prediction using Random Forest Regression. It demonstrates data preprocessing, feature engineering, and model building with scikit-learn. The project aims to provide an example of machine learning applied to real estate datasets.

# House Price Prediction with Random Forest Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-brightgreen)
![pandas](https://img.shields.io/badge/pandas-1.3.3-blueviolet)

Welcome to the House Price Prediction repository! This repository contains Python code to predict house prices using Random Forest Regression. We'll preprocess the data, build the model, and evaluate its performance using Root-Mean-Squared-Error (RMSE).

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Contributing](#contributing)


## Introduction

In this project, we use the Random Forest Regression algorithm from scikit-learn to predict house prices based on provided features. We'll preprocess the data, handle missing values, and use a ColumnTransformer to handle both numerical and categorical features.

## Requirements

- Python 3.8 or higher
- scikit-learn 0.24.2
- pandas 1.3.3

## Usage

1. Clone the repository to your local machine:
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

markdown
Copy code

2. Install the required dependencies:
pip install -r requirements.txt

markdown
Copy code

3. Download the dataset files (`train.csv` and `test.csv`) and place them in the same directory as the Python script.

4. Run the prediction script:
python house_price_prediction.py

markdown
Copy code

5. The script will perform data preprocessing, build the Random Forest Regression model, and make predictions on the test data. The results will be saved in a `submission.csv` file.

## Data

The dataset includes the following files:

- `train.csv`: The training data with features and target variable (SalePrice).
- `test.csv`: The test data used for making predictions.

## Model

The Random Forest Regression model is used for predicting house prices. You can experiment with different regression algorithms available in scikit-learn by changing the `model` variable.

## Contributing

We welcome contributions to this repository. If you find any issues or want to add new features, please feel free to open an issue or submit a pull request.

