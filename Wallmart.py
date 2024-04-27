'''import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/walmart-recruiting-store-sales-forecasting/train.csv/train.csv")
data = df.to_numpy()

# Handling missing values
print(df.isnull().sum())

# Plot boxplot before handling missing values
def plot_boxplot(data, column):
    sns.boxplot(y=column, data=data)
    plt.title(f"Boxplot of {column}")
    plt.show()

graphs = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
for column in graphs:
    plot_boxplot(df, column)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# One-hot encode 'Date' column
encoded_df = pd.get_dummies(df, columns=['Date'])

# Prepare data for linear regression
X = encoded_df.drop('Weekly_Sales', axis=1)
y = encoded_df['Weekly_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Train the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)'''


'''import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.metrics import mean_squared_error

# Generate example data
true_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.2, 1.8, 2.9, 3.7, 5.2])

# Compute mean squared error
mse = mean_squared_error(true_values, predicted_values)
print("Mean Squared Error:", mse)

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/walmart-recruiting-store-sales-forecasting/train.csv/train.csv")
data = df.to_numpy()'''

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
df = pd.read_csv("C:/Users/DELL/Downloads/walmart-recruiting-store-sales-forecasting/train.csv/train.csv")
data = df.to_numpy()

# Handling missing values
print(df.isnull().sum())

# Plot boxplot before handling missing values
def plot_boxplot(data, column):
    sns.boxplot(y=column, data=data)
    plt.title(f"Boxplot of {column}")
    plt.show()

graphs = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
for column in graphs:
    plot_boxplot(df, column)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# One-hot encode 'Date' column
encoded_df = pd.get_dummies(df, columns=['Date'])

# Prepare data for linear regression
X = encoded_df.drop('Weekly_Sales', axis=1)
y = encoded_df['Weekly_Sales']

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