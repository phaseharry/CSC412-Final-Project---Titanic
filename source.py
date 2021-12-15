# Import the modules needed
# pandas is used to load and manipulate data and for One-Hot Encoding
import pandas as pd
# Import the modules needed
import numpy as np # data manipulation
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample # downsample the dataset
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.preprocessing import scale # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.metrics import confusion_matrix # this creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.decomposition import PCA # to perform PCA to plot the data
from sklearn import preprocessing

#For decision tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LogisticRegression #For Logistic Regression
from sklearn.neighbors import KNeighborsClassifier #For K Nearest Neighbors
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('titanic.csv', header=0)

print(df.head())
print(len(df)) # 891 records in total

df.drop('PassengerId', axis=1, inplace=True) # removing "PassengerId" as a column from the data
print(df.head())

columns = [
    'Survived',
    'Pclass',
    'Name',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'Embarked'
]

# looking at all unique values for all columns (skipping Name)
for key in columns:
    if key == 'Name':
        continue
    print("Unique values for " + key + ": ")
    print(df[key].unique())
    print("\n\n")

#Booleans
label = preprocessing.LabelEncoder();
df['Embarked'] = label.fit_transform(df['Embarked']) #Boolean for whether embarked or not
df['Sex'] = label.fit_transform(df['Sex']) #Boolean for whether male or not
df['Cabin'] = label.fit_transform(df['Cabin']) #Boolean for whether or not someone has a cabin

#Fill in missing ages with 100
df['Age'].fillna(100,inplace=True)

#Set up X and y
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
y = df['Survived']

#Set up training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.2, random_state = 0)

#Now we need to apply 3 different models to this data and see their accuracy scores

#First we apply Decision Tree (Model 1)
model1 = tree.DecisionTreeClassifier().fit(X.values, y.values)
y_pred = model1.predict(X_test)

#Plot Decision Tree (Model 1)
tree.plot_tree(model1)
plt.show()

#Get accuracy score for Decision Tree (Model 1)
print("Decision Tree Accuracy: ", accuracy_score(y_test, y_pred))

#Next we apply Logistic Regression (Model 2)
model2 = LogisticRegression(max_iter = 2000).fit(X_train, y_train)
y_pred = model2.predict(X_test)

#Get accuracy score for Logistic Regression (Model 2)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))

#ROC Curve and AUC Score for Logistic Regression Model
y_pred_proba = model2.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data, AUC = " + str(auc))
plt.legend(loc = 4)
plt.show()

#Finally we apply K-Nearest Neighbors (Model 3)
model3 = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
y_pred = model3.predict(X_test)

#Get accuracy score for K-Nearest Neighbors (Model 3)
print("K-Nearest Neighbors Accuracy: ", accuracy_score(y_test, y_pred))

#Use cross validation and regularization to try to improve regression accurracy
#Trying regularization for Logistic Regression
model2part2 = LogisticRegression(max_iter = 2000, penalty = 'l2').fit(X_train,y_train)
y_pred = model2part2.predict(X_test)
print('Logistic Regression Accuracy with Regularization: ', accuracy_score(y_test, y_pred))
#Trying cross validation
model2part3 = LogisticRegression(max_iter = 2000)
valscores = cross_val_score(model2part3, X, y, cv = 5)
print('Cross Validation Scores: ', valscores)