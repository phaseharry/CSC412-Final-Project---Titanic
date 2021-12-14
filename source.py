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
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv', header=0)

print(df.head())
print(len(df)) # 891 records in total

# - need to turn sex into 0 or 1, 0 for male, 1 for female
# - could remove PassengerId as it will liekly not play a role
# - some poeple have missing ages (remove from pool?), will depend on how many samples are missing out of the whole. Initial instructions are to put 100 for all missing at the moment

#Not sure if we have to remove this yet
#df.drop('PassengerId', axis=1, inplace=True) # removing "PassengerId" as a column from the data
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
X_train, X_test, y_train, t_test = train_test_split(X.values, y.values, test_size = 0.2, random_state = 0)

#Now we need to apply 3 different models to this data and see their accuracy scores