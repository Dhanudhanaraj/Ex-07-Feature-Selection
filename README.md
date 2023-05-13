# Ex-07-Feature-Selection
## AIM

To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation

Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM

### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature selection techniques to all the features of the data set

### STEP 4

Save the data to the file

# CODE

NAME:DHANUMALYA.D

REGISTER NUMBER:212222230030
```
# DATA PREPROCESSING BEFORE FEATURE SELECTION:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()

#checking data
df.isnull().sum()

#removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()

#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()

from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 

# FEATURE SELECTION:
# FILTER METHOD:
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

# HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

# BACKWARD ELIMINATION:
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# RFE (RECURSIVE FEATURE ELIMINATION):
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

# OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# FINAL SET OF FEATURE:
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# EMBEDDED METHOD:
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
 ```
## OUTPUT

### DATA PREPROCESSING BEFORE FEATURE SELECTION:

![DS71](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/05254188-b035-47c5-9145-677973679b04)

![DS72](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/80c14213-2e28-4243-b12a-27550259f7fc)

![DS73](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/ebb654d0-1e27-4c08-8ebd-1ba2d34b5354)

![DS74](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/6076437d-4ace-4b99-b753-13ac3f0a0ddd)

![DS75](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/93aa5635-249f-4a95-a27f-8acee8a7a480)

![DS76](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/89c7e740-fd92-401e-a52f-f23cce52d8fc)

![DS77](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/2f9d9f66-fbe7-4125-a43d-199023bae177)

### FEATURE SELECTION:

#### FILTER METHOD:

The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.

![DS78FS](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/c4a00e72-030f-4c24-80cf-59b552acb1cd)


#### HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:

![DS779](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/6a34f878-598d-4c69-a640-a44b6a8bca9f)

#### WRAPPER METHOD:

Wrapper Method is an iterative and computationally expensive process but it is more accurate than the filter method.

There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.

#### BACKWARD ELIMINATION:

![DS710](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/7e6fed51-9662-4b40-a021-2b4a9463ad86)

#### RFE (RECURSIVE FEATURE ELIMINATION):

![DS711](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/316ee380-f77d-42af-b2ba-8f8c178e1b95)

#### OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:

![DS712](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/38619ffa-4f8a-4eaf-8b20-52950dad6546)

#### FINAL SET OF FEATURE:

![DS713](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/0a2c4fa2-f64e-4a2b-94d8-033487b9eb81)

#### EMBEDDED METHOD:

Embedded methods are iterative in a sense that takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration. Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

![DS714](https://github.com/Dhanudhanaraj/Ex-07-Feature-Selection/assets/119218812/2f2c6ced-bd3b-48a9-80f6-eca58dba691f)

## RESULT:

Thus, the various feature selection techniques have been performed on a given dataset successfully.
