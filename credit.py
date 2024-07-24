import pandas as pd
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv("D:/Disha/Diploma/AI ML/Fraud Detection/creditcard.csv")

# Assuming the last column contains labels (fraud or not fraud)
# Adjust this if your dataset structure is different
X = data.iloc[:, :-1]  # Features
y_true = data.iloc[:, -1]  # True labels

# Assuming you have a model that predicts the labels
# Replace predicted_labels with your actual predicted labels
# Here, it's just assumed to be all zeros for demonstration
y_pred = pd.Series([0] * len(data))

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:",accuracy)