import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# File path
file_path = r"D:/Disha/Diploma/AI ML/boston.csv"

# Check if the file exists
import os
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the dataset
df = pd.read_csv(file_path)

# Check for missing values
print(df.isnull().sum())

# Fill missing values if any (assuming 'CRIM' was a placeholder, this will be extended to other columns if needed)
df['CRIM'].fillna((df['CRIM'].mean()), inplace=True)
print(df.isnull().sum())

# Prepare the data
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Checking outliers and replacing them (simplified and corrected approach)
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: upper if x > upper else lower if x < lower else x)

for col in df.columns:
    if col != 'MEDV':  # Avoid transforming the target variable
        replace_outliers(df, col)
        sns.boxplot(y=col, data=df)
        plt.title(f"Box Plot of {col}")
        plt.show()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Linear Regression
reg = LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

# Evaluation
score = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", score)
