"""
Created on Thu May  7 12:32:38 2020
@author: DESHMUKH
DECISION TREE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_column',None)

# =================================================================================
# Business Problem - Use Decision Trees to prepare a model on fraud data.
# =================================================================================

fraud = pd.read_csv("Fraud_check.csv")
fraud.head()
fraud.isnull().sum()

# Summary 
fraud.describe()

# Boxplot
fraud.boxplot(notch=True, patch_artist=True,grid=False)

# Histogram
fraud.hist(grid=False)

# Output variable Histogram
plt.hist(fraud['Taxable.Income'])

# Cheking Distribution above 30000 and below 30000.
sum(fraud['Taxable.Income'] >= 30000)/len(fraud) # 79% observations are less than 30000
sum(fraud['Taxable.Income'] < 30000)/len(fraud) # 21% observations are less than 30000

# Coverting Fraud data into Categorical form, Treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
fraud['Taxable.Income'] = pd.cut(fraud['Taxable.Income'], bins=[0,30000,100000], labels={ 'Risky','Good'})

# Checking Percentage of Output classes with the help Value Count.
(fraud['Taxable.Income'].value_counts())/len(fraud)*100 # Risky = 79% , Good = 21% Same as before convertion.(so value converted carrectly)

################################ - Converting into Numeric - ################################

fraud.head()
ln = LabelEncoder()
fraud['Undergrad'] = ln.fit_transform(fraud['Undergrad']) 
fraud['Marital.Status'] = ln.fit_transform(fraud['Marital.Status']) 
fraud['Urban'] = ln.fit_transform(fraud['Urban']) 

fraud.head()

################################ - Spliting data in X and y - ################################

X = fraud.drop(['Taxable.Income'],axis = 1)
y = fraud['Taxable.Income']

############################# - Spliting data in train and test - #############################

# Stratified random sampling becuase output have inbalanced Data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30, random_state = False, stratify = y )

# Rechecking Percentage of Output classes by using values counts. 
(y_train.value_counts()/len(y_train))*100 # Percentage of sample Output classes simillier to population 

################################## - Decision Tree Model - ###################################

regressor = DecisionTreeClassifier(criterion = 'entropy')       # Another common use criterio is Gini Index
regressor.fit(X_train,y_train)          

# Prediction on Train & Test Data
pred_train = regressor.predict(X_train)
pred_test = regressor.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,pred_train) 
accuracy_score(y_test,pred_test) 

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,pred_train,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,pred_test,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Reds',fmt='g')

# Classification Report of test
print(classification_report(y_test,pred_test))

################################# - Visualizing Decision Trees - ###################################

#from sklearn.tree import plot_tree
plt.figure(figsize=(25,10))
a = plot_tree(regressor, 
              feature_names = X_train.columns, 
              class_names = fraud.columns[2], 
              filled=True, 
              rounded=True,#fontsize=14
              )

                         # ---------------------------------------------------- #

