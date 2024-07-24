


## Black Friday

The code preprocesses a dataset and trains a Linear Regression model to predict purchases. It begins by loading the data, filling missing values, and encoding categorical variables. Ordinal features are mapped to numerical values, and numerical columns are standardized. Features are defined by excluding 'Purchase' and 'User_ID', while 'Purchase' is set as the target variable. The data is split into training and testing sets, and a Linear Regression model is trained on the training set. Predictions are made on the test set, and the model's performance is evaluated using the Mean Squared Error (MSE).


## About project
1: Load Data: Read the dataset from a CSV file into a pandas       DataFrame.

2: Check Missing Values: Identify and print the count of missing values for each column.

3: Handle Missing Values: Fill missing values in 'Product_Category_1', 'Product_Category_2', and 'Product_Category_3' with 0.

4: Encode Categorical Variables: Label encode 'Product_ID' and 'Gender', and apply one-hot encoding to 'City_Category'.

5: Map Ordinal Features: Convert 'Age' and 'Stay_In_Current_City_Years' to numerical values using predefined mappings.

6: Feature Scaling: Standardize numerical columns using StandardScaler.

7: Define Features and Target: Define features (X) by excluding 'Purchase' and 'User_ID', and set 'Purchase' as the target variable (y).

8: Split Data: Split the dataset into training and testing sets with an 80-20 ratio.

9: Train Model: Initialize and train a LinearRegression model using the training data.

10: Predict and Evaluate: Predict purchases on the test set and evaluate the model using Mean Squared Error (MSE).
## Installation

To run the provided code, you need to install several Python libraries. Here are the required installations:

1: Pandas - For data manipulation and analysis.

2: Sklearn - For machine learning tools, including preprocessing, model training, and evaluation.
## Result

The project give output like this:

1: Mean Square Error:

    Mean Squared Error: 0.8455185912138407