'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
loan_data = pd.read_csv("C:/Users/DELL/Downloads/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")

# Check and handle missing values if necessary
# For simplicity, let's drop rows with missing values
loan_data.dropna(inplace=True)

# Perform one-hot encoding for categorical variables
loan_data_encoded = pd.get_dummies(loan_data, drop_first=True)

# Split data into features (X) and target variable (y)
X = loan_data_encoded.drop(columns=["Loan_Status_Y"])  # Features
y = loan_data_encoded["Loan_Status_Y"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Convert predicted values to binary classes (0 or 1)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)'''





import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
loan_data = pd.read_csv("C:/Users/DELL/Downloads/Loan Prediction/train_u6lujuX_CVtuZ9i.csv")

# Check and handle missing values if necessary
# For simplicity, let's drop rows with missing values
loan_data.dropna(inplace=True)

# Perform one-hot encoding for categorical variables
loan_data_encoded = pd.get_dummies(loan_data, drop_first=True)

# Split data into features (X) and target variable (y)
X = loan_data_encoded.drop(columns=["Loan_Status_Y"])  # Features
y = loan_data_encoded["Loan_Status_Y"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Train the model with the best parameters
rf_regressor_best = grid_search.best_estimator_
rf_regressor_best.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor_best.predict(X_test)

# Convert predicted values to binary classes (0 or 1)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Feature Importance
feature_importance = rf_regressor_best.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


