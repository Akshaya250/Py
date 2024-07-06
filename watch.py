'''The customer which have current account balance of above 30000 and transaction amount
above 5000 can buy the products and the age is in between 18 to 40.'''

import pandas as pd

df = pd.read_csv("D:/Disha/Diploma/AI ML/Data Analytics - BankData/bank_transactions.csv")

missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())


categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

missing_values_after_imputation = df.isnull().sum()
print("Missing values in the dataset after imputation:")
print(missing_values_after_imputation)

filtered_df = df[(df['CustAccountBalance'] > 30000) & (df['TransactionAmount (INR)'] > 5000)]

rows_to_drop = df[df['CustomerDOB'] == '1/1/1800'].index
df.drop(rows_to_drop, inplace=True)

total_customers = len(filtered_df)
filtered_df = filtered_df[(filtered_df['CustomerDOB'] >= '1980-11-02') & (filtered_df['CustomerDOB'] <= '2005-01-05')]
print("People with bank balance more than 40000 and transaction amount more than 20000:")
print(filtered_df)
print("\nTotal number of customers: ", total_customers)
