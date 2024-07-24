
# Company Bankrupt

The code loads a dataset for bankruptcy prediction, handles missing values, balances class distribution using RandomOverSampler, splits data into training and testing sets, trains a Random Forest classifier, predicts bankruptcy status, and evaluates model performance using Mean Square Error and Accuracy.








## About Project
1:Objective: The project aims to develop a predictive model to assess the likelihood of bankruptcy for companies based on financial indicators.

2: Dataset: The project utilizes a dataset containing various financial features of companies, such as profitability, liquidity, and solvency ratios, along with a binary target variable indicating bankruptcy status.

3: Data Exploration: Exploratory data analysis (EDA) is conducted to understand the structure of the dataset, identify missing values, and gain insights into the distribution of features and the target variable.

4: Data Preprocessing: Missing values are handled appropriately, and the dataset is prepared for model training by separating features from the target variable.

5: Class Imbalance Handling: Given the potential class imbalance between bankrupt and non-bankrupt companies, oversampling techniques like RandomOverSampler are employed to ensure balanced class distribution in the training data.

6: Model Selection: A Random Forest classifier is chosen as the predictive model due to its ability to handle complex relationships between features and its robustness against overfitting.

7: Model Training: The Random Forest classifier is trained on the resampled training data to learn patterns and relationships between financial indicators and bankruptcy status.

8: Model Evaluation: The trained model is evaluated using standard evaluation metrics such as Mean Square Error and Accuracy to assess its performance in predicting bankruptcy status on unseen data.

9: Performance Tuning: Hyperparameter tuning techniques may be applied to optimize the performance of the Random Forest classifier and improve its predictive accuracy.

10: Deployment and Monitoring: Once the model achieves satisfactory performance, it can be deployed in a production environment for real-time prediction of bankruptcy risk. Continuous monitoring and periodic model retraining may be necessary to ensure its effectiveness over time.
## Installation
1:Pandas: Install using pip install pandas.

2: Imbalanced-learn: Install using pip install imbalanced-learn.

3: Scikit-learn: Install using pip install scikit-learn.

Ensure these packages are installed in your Python environment to run the script successfully.
## Result
The project give output like this:

1: Mean Square Error:

    Mean Square Error: 0.003409090909090909

2: Accuracy:

    Accuracy: 0.9965909090909091