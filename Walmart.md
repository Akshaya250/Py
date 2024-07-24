
# Walmart

The script begins by calculating the mean squared error between true and predicted values as a demonstration. Then, it loads a dataset and checks for missing values, plotting boxplots before handling them. It converts the 'Date' column to datetime format and one-hot encodes it. Next, it prepares the data for linear regression by splitting it into training and testing sets. A linear regression model is trained on both the original and scaled features using StandardScaler. Predictions are made on the test set, and the mean squared error is computed for the scaled predictions. Finally, the script prints the mean squared error for the scaled predictions.


## About project
1: Objective: The project aims to build a regression model to predict weekly sales based on various features.

2: Dataset: Utilizes a dataset (train Wal.csv) containing information such as store, department, date, weekly sales, and holiday status.

3: Data Preprocessing: Handles missing values by identifying and addressing them appropriately. Converts the 'Date' column to datetime format and one-hot encodes it for better model compatibility.

4: Exploratory Data Analysis (EDA): Visualizes the distribution of data using boxplots to understand feature characteristics and potential outliers.

5: Feature Engineering: Prepares the data for regression by splitting it into features (X) and the target variable (y).

6: Model Training: Utilizes Linear Regression as the predictive model due to its simplicity and interpretability.

7: Model Evaluation: Splits the data into training and testing sets to evaluate the model's performance. Computes Mean Squared Error (MSE) to assess the accuracy of the model's predictions.

8: Feature Scaling: Scales the features using StandardScaler to ensure uniformity and improve model convergence.

9: Performance Comparison: Compares the model's performance on original and scaled features by calculating MSE for both cases.

10: Conclusion: Provides insights into the model's predictive capabilities and highlights the importance of feature scaling in improving model performance.





## Installation 
1: Python 3.x: The programming language used for developing the project.

2: Pandas: For data manipulation and analysis.

3: NumPy: For numerical operations and array manipulation.

4: Scikit-learn: For implementing machine learning models and evaluation metrics.

5: Seaborn: For statistical data visualization.

6: Matplotlib: For creating various plots and visualizations.
## Result
It will give result like this:

MSA:

    Mean Squared Error (scaled features): 499164717.7780505