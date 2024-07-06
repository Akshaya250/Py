
# Titanic 

The script analyzes the Titanic dataset, aiming to predict the 'Embarked' port of passengers using a Random Forest classifier. It begins by reading the dataset and preprocessing it, including handling missing values and converting categorical variables to numerical ones using Label Encoding. Outliers in numerical features are addressed using a function to handle outliers. The data is then split into training and testing sets. After training the Random Forest classifier on the training data, predictions are made on the test set. Finally, the script evaluates the model's performance using mean squared error and accuracy metrics.


## About project
1: Objective: The project aims to predict the port of embarkation ('Embarked') for Titanic passengers using machine learning techniques.

2: Data Source: The Titanic dataset is utilized, obtained from a CSV file containing passenger information.

3: Data Preprocessing: Missing values in the 'Age' and 'Embarked' columns are filled with appropriate values, and outliers in numerical features are handled using a custom function.

4: Feature Encoding: Categorical variables like 'Sex' and 'Ticket' are encoded using Label Encoding to convert them into numerical format.

5: Feature Selection: Although not explicitly mentioned, feature selection techniques could be applied to identify the most relevant features for model training.

6: Model Selection: A Random Forest classifier is chosen as the predictive model for its ability to handle categorical data and potential interactions between features.

7: Model Training: The Random Forest classifier is trained on the training data split from the dataset.

8: Model Evaluation: The performance of the trained model is evaluated using mean squared error and accuracy metrics to assess its predictive accuracy.

9: Testing and Predictions: The trained model is used to make predictions on the test dataset to determine the accuracy of the model's predictions.

10: Analysis: The project concludes with an analysis of the model's performance, highlighting its accuracy in predicting the port of embarkation for Titanic passengers.





## Installation 
1: Pandas: For data manipulation and analysis.

2: Matplotlib: For data visualization.

3: Seaborn: For statistical data visualization.

4: Sklearn: For machine learning algorithms and model evaluation.
## Result
The output will be like this:

    Mean Squared Error: 0.30726256983240224

    Accuracy of Random Tree: 0.9106145251396648