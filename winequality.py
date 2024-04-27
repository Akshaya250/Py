'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Wine Quality/winequalityN.csv")

# Drop any rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Split features and target variable
X = df.drop(columns=["quality"])
y = df["alcohol"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)'''


'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Wine Quality/winequalityN.csv")

# Split features and target variable
X = df.drop(columns=["alcohol"])
y = df["quality"]

# Encode categorical variables
label_encoder = LabelEncoder()
X['type'] = label_encoder.fit_transform(X['type'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize and train the Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train_imputed, y_train)

# Make predictions
y_pred = clf.predict(X_test_imputed)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)'''


'''import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import mean_squared_error

# Generate example data
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

# Compute mean squared error
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error:", mse)

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Wine Quality/winequalityN.csv")
data = df.to_numpy()

# Handling missing values
print(df.isnull().sum())

# Plot boxplot before handling missing values
def plot_boxplot(data, column):
    sns.boxplot(y=column, data=data)
    plt.title(f"Boxplot of {column}")
    plt.show()

graphs = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides']
for column in graphs:
    plot_boxplot(df, column)

# Convert 'Date' column to datetime
df['quality'] = pd.to_datetime(df['quality'])

# One-hot encode 'Date' column
encoded_df = pd.get_dummies(df, columns=['quality'])

# Prepare data for linear regression
X = encoded_df.drop('fixed acidity', axis=1)
y = encoded_df['fixed acidity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Train the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model on scaled features
reg.fit(X_train_scaled, y_train)

# Predictions on scaled features
y_pred_scaled = reg.predict(X_test_scaled)

# Calculate Mean Squared Error on scaled predictions
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print("Mean Squared Error (scaled features):", mse_scaled)'''



'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Wine Quality/winequalityN.csv")

# Split features and target variable
X = df.drop(columns=["alcohol"])
y = df["quality"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical variables using one-hot encoding
X_train = pd.get_dummies(X_train, columns=['type'])
X_test = pd.get_dummies(X_test, columns=['type'])

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import numpy as np
from sklearn.metrics import mean_squared_error

# Generate example data
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

# Compute mean squared error
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error:", mse)'''