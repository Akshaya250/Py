
# Boston (ML Regression)

The project analyzes the Boston housing dataset to predict the median value of owner-occupied homes ('MEDV') using a Linear Regression model. It begins by reading the dataset and checking for missing values, which are then filled with appropriate statistics. The project involves extensive outlier detection and handling for multiple features using the interquartile range (IQR) method. Label Encoding is applied to categorical variables. The data is split into training and testing sets, and a Linear Regression model is trained on the training data. The model's performance is evaluated on the test set using the mean squared error metric. Visualization of data distributions and outliers is performed using Seaborn and Matplotlib.


## About project
1: Objective: Predict the median value of owner-occupied homes ('MEDV') in the Boston housing dataset using Linear Regression.

2: Data Source: Uses the Boston housing dataset, read from a CSV file.

3: Missing Value Handling: Identifies and fills missing values in the dataset with the mean value.

4: Feature Encoding: Converts categorical features to numerical values using Label Encoding.

5: Outlier Detection: Detects outliers in various features using the interquartile range (IQR) method.

6: Outlier Handling: Replaces outliers in features with the calculated upper and lower bounds based on IQR.

7: Data Visualization: Uses Seaborn and Matplotlib to create box plots for visualizing the distributions and outliers of features.

8: Feature Selection: Selects all features except 'MEDV' for the input (X) and 'MEDV' as the target (y).

9: Model Training: Splits the dataset into training and testing sets, then trains a Linear Regression model on the training data.

10: Model Evaluation: Predicts on the test set and evaluates model performance using mean squared error, providing an accuracy metric for the model's predictions.
## Installation 
1: Pandas: For data manipulation and analysis.

2: Sklearn: For machine learning algorithms and model evaluation.

3: Matplotlib: For data visualization.

4: Seaborn: For statistical data visualization.
## Result
The output will be like this:

    Mean Squared Error: 23.380836480270304