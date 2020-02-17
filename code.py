# --------------
# Code starts here

#### Data 1
# Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Overview of the data
df= pd.read_csv(path)
df.info()
df.describe()
# Histogram showing distribution of car prices
ax1 = plt.hist(df['price'], bins=30)
plt.title('Distribution of car prices')
plt.show()
# Countplot of the make column
ax2 = sns.countplot(x='make', data=df)
plt.title ('Count of Make')
plt.show()
# Jointplot showing relationship between 'horsepower' and 'price' of the car
ax3 = sns.jointplot(x='horsepower', y='price', data=df)
plt.title('Relationship between horsepower and price')
plt.show()
# Correlation heat map
cordata =  df.corr()
ax4 = sns.heatmap(cordata)
plt.title ('HeatMap')
plt.show()
# boxplot that shows the variability of each 'body-style' with respect to the 'price'
ax5 = sns.boxplot(x='body-style', y='price', data=df)
plt.title('Variability of body style with Price')
plt.show()
#### Data 2
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
# Load the data
df1 = pd.read_csv(path2)
print(df1.head())

# Impute missing values with mean
df1['normalized-losses']=df1['normalized-losses'].str.replace("?","NaN")
df1['horsepower']=df1['horsepower'].str.replace("?","NaN")

mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

mean_imputer.fit(df1[['normalized-losses']])
df1['normalized-losses'] = mean_imputer.transform(df1[['normalized-losses']])

mean_imputer.fit(df1[['horsepower']])
df1['horsepower'] = mean_imputer.transform(df1[['horsepower']])

# Skewness of numeric features
df1.info()
print(df1.skew(axis=0, skipna=True))

a1=plt.hist(df1['engine-size'], bins=20)
plt.show()
a2 = plt.hist(df1['horsepower'], bins=20)
plt.show()

df1['engine-size'] = np.sqrt(df1['engine-size'])
df1['horsepower'] = np.sqrt(df1['horsepower'])

# Label encode 
le=LabelEncoder()
cat_col = ['make','fuel-type','body-style','drive-wheels','engine-location', 'engine-type']
for i in cat_col:
    df1[i] = le.fit_transform(df1[i])

df1['area'] = df1['height'] * df1['width']
print(df1.head())

# Code ends here


