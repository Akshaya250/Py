
# Watch

Your script processes bank transaction data to identify customers eligible to buy products based on specific criteria: a current account balance above 30,000, a transaction amount above 5,000, and an age between 18 and 40 years. The script reads the data, handles missing values by filling numerical columns with the mean and categorical columns with the mode, and filters out invalid birthdates. It then filters customers who meet the criteria and calculates their ages correctly. Finally, it outputs the filtered data and the total number of eligible customers.








## About project
1: The project identifies eligible customers based on financial and age criteria.

2: The dataset used is a CSV file containing bank transaction data.

3: Criteria for eligibility include an account balance above 30,000, transaction amount above 5,000, and age between 18 and 40.

4: Missing numerical values are imputed with the mean.

5: Missing categorical values are imputed with the mode.

6: Rows with invalid birthdates ('1/1/1800') are removed.

7: 'CustomerDOB' is converted to datetime format for accurate age calculation.

8: Customers are filtered based on the specified financial and age criteria.

9: The filtered dataset and the total number of eligible customers are displayed.

10: The project ensures data quality and provides insights into the eligible customer base.
## Installation 
1: Pandas: For data manipulation and analysis.

2:NumPy: To handle numerical data and missing values efficiently.

## Result
The output will be like this:

    Total number of customers:  33724