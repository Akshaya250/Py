'''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

logr=LogisticRegression()

df = pd.read_csv("C:/Users/DELL/Downloads/archive (1)/Iris.csv")

x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred)'''

'''import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nb=GaussianNB()
df = pd.read_csv("C:/Users/DELL/Downloads/archive (1)/Iris.csv")
x = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)
nb.fit(X_train,y_train)

y_pred1=nb.predict(X_test)

print("Naive Bayes: ", accuracy_score(y_test,y_pred1))'''

'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("C:/Users/DELL/Downloads/archive (1)/Iris.csv")
x = df.drop(['Id', 'Species'], axis=1)
y = df['Species']
knn=KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
train=knn.fit(X_train,y_train)

y_predd=knn.predict(X_test)

print(accuracy_score(y_test,y_predd))'''

'''from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dt=tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
train=dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)'''

'''from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

df = pd.read_csv("C:/Users/DELL/Downloads/archive (1)/Iris.csv")

x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']

gbm=GradientBoostingClassifier(n_estimators=10)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
gbm.fit(X_train, y_train)
y_pred=gbm.predict(X_test)
print("GBM: ", accuracy_score(y_test,y_pred))
'''
