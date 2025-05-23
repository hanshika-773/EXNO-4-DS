# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
from google.colab import drive
drive.mount('/content/drive')
```
## Feature Scaling
```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
df.head()
```
![437925219-76f41d86-97e6-4b87-b33c-110004284747](https://github.com/user-attachments/assets/d43f61ad-d7cf-4826-a912-c33cc12dea6f)
```
df.dropna()
```
![437925246-3dac35be-b8ee-4b2f-8b64-5b2535f549f0](https://github.com/user-attachments/assets/cf2149d1-be89-422b-8904-90fbf4ed73ab)
```
print("Max Height:", df['Height'].max())
print("Max Weight:", df['Weight'].max())
```
![437925271-faeeca5d-1f4a-4704-9593-a018285cf43e](https://github.com/user-attachments/assets/f59e5372-6106-4e9e-ac60-621db8f7fe87)
```
   from sklearn.preprocessing import MinMaxScaler
   scaler=MinMaxScaler()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df.head(10)
```
![437925331-b261c234-5f3e-42b1-96ec-d1782f18c27a](https://github.com/user-attachments/assets/636a3662-cc8a-42cc-a094-8a26db5f6a98)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![437925408-d81e5cff-4aeb-4ad7-834b-00b1b0709fb2](https://github.com/user-attachments/assets/0e962ed0-658d-4d42-b934-0c1a82da5664)
```
   from sklearn.preprocessing import Normalizer
   scaler=Normalizer()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df
```
![437925447-a364d8ec-2ba2-4655-ba5e-30d4c9590b6f](https://github.com/user-attachments/assets/d801e743-047d-49d5-a1c1-56ef0e47e959)
```
   df1=pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
   from sklearn.preprocessing import MaxAbsScaler
   scaler=MaxAbsScaler()
   df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
   df1
```
![437925462-3de37039-1943-4787-a446-be80c25e94bd](https://github.com/user-attachments/assets/3617317f-650b-444b-8a4f-0d6c0b12e126)
```
   df2=pd.read_csv("/content/drive/MyDrive/Data_Science/bmi.csv")
   from sklearn.preprocessing import RobustScaler
   scaler=RobustScaler()
   df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   df2.head()
```
![437925497-9a343dbc-071a-4c0e-b699-180e6c3a3d47](https://github.com/user-attachments/assets/5f0930fe-1f2a-4556-8c5c-face5b6f4488)
## Feature Selection
```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv('/content/drive/MyDrive/Data_Science/titanic_dataset.csv')
df.columns
```
![437925552-2de662f1-0464-4e26-97c5-9468650d9b18](https://github.com/user-attachments/assets/82759228-5c06-42c2-af26-efd0f3872c4f)
```
df.shape
```
![437925579-1fd1be08-9714-494a-9ad3-9d2d383636c2](https://github.com/user-attachments/assets/be3cef8e-05b9-4073-af12-afae06a0c004)
```
X = df.drop("Survived", axis=1)       # feature matrix
y = df['Survived']

df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
df1
```
![437925627-f06d7689-a602-494b-9160-b613e37739d7](https://github.com/user-attachments/assets/de316e19-b1ef-4ff9-b764-605a88a9ab26)
```
df1.columns
```
![437925649-98bff8e0-27ea-4f9f-90cc-454ce1b8f749](https://github.com/user-attachments/assets/076e8e09-6f3f-4778-bb81-4aa5c3e3eb83)
```
df1['Age'].isnull().sum()
```
![437925679-fb21ad2d-e868-480f-a5b4-11031f98093d](https://github.com/user-attachments/assets/b787645b-cde5-4fd0-a930-684f071948db)
```
df1['Age'] = df1['Age'].ffill()
df1['Age'].isnull().sum()
```
![437925709-4558b30f-071f-4aa8-bbfc-175938bdd0b6](https://github.com/user-attachments/assets/f17b4f06-a5d0-480c-97ca-7737aa92dd80)
```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![437925745-b98c56fb-528f-42d3-9842-7083dd10c4fb](https://github.com/user-attachments/assets/46586e59-8dc0-4d41-8d25-ca3e04a4c232)
```
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
df1
```
![437925838-2a7e2142-f5aa-4d1b-ab3e-726f57b19610](https://github.com/user-attachments/assets/3a1436c4-9e3e-4fc5-84bc-c312ee629762)
```
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
```
![437925938-e504c7ba-f00d-4e7d-9651-37603a4bad6a](https://github.com/user-attachments/assets/b6eba9da-d246-463e-aad6-75bb805daabd)
```
y=y.to_frame()
y.columns
```
![437926053-ac08abc4-cb20-410d-99c0-e693c0729514](https://github.com/user-attachments/assets/923a9464-f754-4e92-a3d3-b269d5b68537)
```
feature.fit(X,y)
```
![437926151-d98d498c-7954-45b0-9a3d-75e5226c7f7f](https://github.com/user-attachments/assets/6095ea1b-7f7d-45b5-903f-3c5041e83625)



# RESULT:
Thus, successfully read the given data and performed Feature Scaling and Feature Selection process and saved the data to a file.
