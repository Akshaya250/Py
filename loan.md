
# Loan

The script processes a loan prediction dataset by first loading the data and handling missing values through imputation, replacing missing numerical values with the mean and categorical values with the mode. It then drops the irrelevant 'Loan_ID' column and encodes categorical variables into numerical format using LabelEncoder. To address class imbalance, it applies RandomOverSampler to balance the dataset. The data is split into training and testing sets, and a Random Forest classifier is trained on the resampled data. Finally, the model's accuracy is evaluated on the test set and printed.








## About project
1: Project Goal: Predict loan approval status using a machine learning model on a processed dataset.

2: Dataset: Utilize the loan prediction dataset (train_u6lujuX_CVtuZ9i.csv), which includes various features related to loan applications.

3: Missing Value Handling: Identify and impute missing values; numerical columns are filled with the mean, and categorical columns with the mode.

4: Feature Selection: Drop the non-predictive 'Loan_ID' column from the dataset.

5: Label Encoding: Convert categorical variables into numerical format using LabelEncoder for model compatibility.

6: Class Imbalance Handling: Apply RandomOverSampler to balance the dataset, addressing any class imbalances in the target variable.

7: Data Splitting: Split the resampled dataset into training (80%) and testing (20%) sets to evaluate model performance.

8: Model Training: Initialize and train a Random Forest classifier on the training dataset.

9: Model Prediction: Use the trained Random Forest classifier to predict loan approval status on the test dataset.

10: Performance Evaluation: Calculate and print the accuracy of the model on the test dataset to assess its predictive performance.
## Installation
1: Install Python (if not already installed).

2: Install the required Python libraries (pandas, scikit-learn, and imbalanced-learn).

3: Ensure the CSV file is in the correct location.
## Result
The output will be like:

1:Accuracy:

    Accuracy: 0.9112426035502958