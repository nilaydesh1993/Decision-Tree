"""
Created on Wed May  6 14:46:26 2020
@author: DESHMUKH
DECISION TREE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import plot_tree
pd.set_option('display.max_columns',None)

# ===================================================================================================================================
# Business Problem - A cloths manufacturing company is interested to know about the segment or attributes contributing to high sale.
# ===================================================================================================================================

companydata = pd.read_csv("Company_Data.csv")
companydata.head()
companydata.shape
companydata.isnull().sum()

# Summary
companydata.describe()

# Boxplot
companydata.boxplot(notch='True',patch_artist=True,grid=False);plt.xticks(fontsize=6)

# Histogram
companydata.hist()

# Sales Histogram
plt.hist(companydata.Sales)

# Coverting sales data into Categorical form, Taking >10 = High and <10 = Low
companydata.Sales = pd.cut(companydata.Sales, bins = [-1,10,17],labels = {'Low','High'}) # It is use to convert Numerical data into Categorical. 
                                                                                        # [0 to 10] = Low , [10 to 17] = High
# Checking Percentage of Output classes with the help group by.
(companydata.groupby('Sales').size()/len(companydata))*100 

################################ - Converting into Numeric - ################################

companydata.head()

le = LabelEncoder()
companydata.ShelveLoc = le.fit_transform(companydata.ShelveLoc)
companydata.Urban = le.fit_transform(companydata.Urban)
companydata.US = le.fit_transform(companydata.US)

companydata.head()

################################ - Spliting data in X and y - ################################

X = companydata.iloc[:,1:]
y = companydata.iloc[:,0]

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
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Greens',fmt='g')

# Classification Report of test
print(classification_report(y_test,pred_test))

################################## - Visualizing Decision Trees - ###################################

#from sklearn.tree import plot_tree
plt.figure(figsize=(25,10))
a = plot_tree(regressor, 
              feature_names = X_train.columns, 
              class_names = companydata.columns[0], 
              filled=True, 
              rounded=True,#fontsize=14
              )

                         # ---------------------------------------------------- #
                         