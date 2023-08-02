    # Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the training and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Extract the target variable (SalePrice) from the training data
target = train_data["SalePrice"]

# Drop the target variable from the training data and any other irrelevant columns
train_data.drop(columns=["Id", "SalePrice"], inplace=True)

# Combine the training and test data for preprocessing
combined_data = pd.concat([train_data, test_data], axis=0)

# Separate categorical and numerical features
categorical_features = combined_data.select_dtypes(include=["object"]).columns.tolist()
numerical_features = combined_data.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Perform data preprocessing with imputation for missing values
combined_data[numerical_features] = combined_data[numerical_features].fillna(combined_data[numerical_features].median())
combined_data[categorical_features] = combined_data[categorical_features].fillna(combined_data[categorical_features].mode().iloc[0])

# Build the preprocessing pipeline for numerical features
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Build the preprocessing pipeline for categorical features
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first'))  # Use drop='first' to avoid the dummy variable trap
])

# Combine the preprocessing pipelines
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Split the combined data back into training and test sets
X_train = combined_data.iloc[:train_data.shape[0]]
X_test = combined_data.iloc[train_data.shape[0]:]

# Transform the data using the preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Build the Random Forest Regression model (you can try other regression algorithms as well)
model = RandomForestRegressor(n_estimators=1000, random_state=42)

# Perform cross-validation on the model to assess its performance
# Use Root-Mean-Squared-Error (RMSE) as the evaluation metric
cv_scores = cross_val_score(model, X_train_preprocessed, target, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)

print("Cross-Validation RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

# Train the model on the entire training data
model.fit(X_train_preprocessed, target)

# Make predictions on the test data
test_predictions = model.predict(X_test_preprocessed)

# Create a submission file
submission_df = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_predictions})
submission_df.to_csv("submission.csv", index=False)
