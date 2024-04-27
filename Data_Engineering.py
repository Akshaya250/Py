'''import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df=pd.read_csv("C:/Users/DELL/Downloads/archive/HousingData.csv")

bos=load_boston()
reg=LinearRegression()

print(bos.DESCR)

x=bos.data
y=bos.target

print(x)
print(y)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.2)
reg.fit(X_train, Y_train)

# Make predictions on the testing data
y_pred = reg.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)'''

'''import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("C:/Users/DELL/Downloads/archive/HousingData.csv")


bos = load_boston()


X = bos.data
y = bos.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


reg = LinearRegression()
reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)'''

#inear regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/Users/DELL/Downloads/archive/HousingData.csv")
print(df)
df.replace('?',"")
print(df.isnull().sum())

df['CRIM']=pd.cut(df['CRIM'],3,labels=['0','1','2'])
df['ZN']=pd.cut(df['ZN'],3,labels=['0','1','2'])
df['INDUS']=pd.cut(df['INDUS'],3,labels=['0','1','2'])
df['CHAS']=pd.cut(df['CHAS'],3,labels=['0','1','2'])
print(df)

X = df.drop('INDUS', axis =1)
X = X.drop('CHAS', axis=1)
Y = df['CHAS']
print(Y)
le=LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)

sns.boxplot(y='CRIM', data = df)                #x=horizontal and  y=vertical
plt.title("Box plot showing the distribution of sepal length")
plt.show()


Q1 = df['CRIM'] .quantile(0.25)
Q3 = df['CRIM'] .quantile(0.75)

IQR = Q3-Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(lower)
print(upper)

out1=df[df['CRIM'] < lower].values
out2=df[df['CRIM'] > upper].values

df['CRIM'].replace(out1,lower)
df['CRIM'].replace(out2,lower)

sns.boxplot(df["CRIM"])
plt.show()
