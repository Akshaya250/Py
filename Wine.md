
# Wine

The provided code performs a series of steps to preprocess a dataset and train a logistic regression model. It starts by loading a dataset and splitting it into features and target variables. The dataset is then split into training and testing sets. Categorical variables are encoded using one-hot encoding, and missing values are imputed with the mean strategy. The features are scaled, and a logistic regression model is trained on the processed data. Finally, the model's accuracy is evaluated on the test set. Additionally, an example of calculating mean squared error for a set of true and predicted values is included.



## About project
1: Dataset Loading: The project begins by loading a wine quality dataset from a CSV file.

2: Feature and Target Split: Features (X) and target variable (y) are separated, with quality as the target and other variables as features.

3: Train-Test Split: The data is split into training and testing sets using an 80-20 split ratio.

4: Categorical Encoding: Categorical variables are encoded using one-hot encoding to convert them into a numerical format.

5: Missing Value Imputation: Missing values in the dataset are handled using the SimpleImputer, which fills in missing values with the mean of each column.

6: Feature Scaling: The features are standardized using StandardScaler to normalize the data.

7: Model Training: A logistic regression model is initialized and trained using the processed training data.

8: Prediction: The trained model is used to make predictions on the test set.

9: Accuracy Calculation: The accuracy of the model is computed to evaluate its performance on the test data.

10: Mean Squared Error Example: A separate example demonstrates the calculation of mean squared error between a set of true and predicted values to illustrate error measurement.
## Installation
1: pandas: For data manipulation and analysis.

2: Sklearn: For machine learning algorithms and preprocessing tools.

3: numpy: For numerical operations.
## Result
The output will be like this:

    Accuracy: 0.9976923076923077
    Mean Squared Error: 0.04399999999999999