
# Iris (ML-Classification)

The code loads the Iris dataset and prepares it by separating features and labels, then splits the data into training and testing sets. It trains six different classifiers: K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Gradient Boosting Machine (GBM), Naive Bayes, and Logistic Regression. Each classifier's accuracy is calculated on the test set. Additionally, for Logistic Regression, the code provides a detailed classification report and confusion matrix to evaluate its performance. This approach allows comparison of different models to determine which one performs best on the Iris dataset.








## About project
1: Data Loading: The Iris dataset is loaded from a CSV file using the pandas library.

2: Data Preparation: The 'Id' and 'Species' columns are dropped from the features set, and 'Species' is used as the target variable.

3: Data Splitting: The dataset is split into training (70%) and testing (30%) sets using the train_test_split method from scikit-learn.

4: Model Training - KNN: A K-Nearest Neighbors classifier with 5 neighbors is trained on the training set.

5: Model Training - Decision Tree: A Decision Tree classifier is trained on the training set.

6: Model Training - Random Forest: A Random Forest classifier is trained on the training set.

7: Model Training - Gradient Boosting: A Gradient Boosting Machine classifier with 10 estimators is trained on the training set.

8: Model Training - Naive Bayes: A Gaussian Naive Bayes classifier is trained on the training set.

9: Model Training - Logistic Regression: A Logistic Regression classifier with a maximum of 200 iterations is trained on the training set.

10: Model Evaluation: Each classifier's accuracy is computed on the test set. For Logistic Regression, additional evaluation is done using a classification report and confusion matrix to provide detailed performance metrics.
## Installation
1: pandas: For data manipulation and analysis.

2: scikit-learn: For machine learning algorithms and model evaluation.
## Result

    KNN Accuracy: 0.9777777777777777
    
    Decision Tree Accuracy: 0.9777777777777777
    
    Random Forest Accuracy: 0.9777777777777777
    
    GBM Accuracy: 0.9777777777777777
    
    Naive Bayes Accuracy: 1.0
    
    Logistic Regression Accuracy: 0.9777777777777777