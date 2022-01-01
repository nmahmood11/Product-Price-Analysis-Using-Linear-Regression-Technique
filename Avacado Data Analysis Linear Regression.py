# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 10:57:50 2022

@author: Noman Mahmood
"""

# Avocado Data Analysis
# Business Understanding
# Explore best answer for required questions
#1. Which region are the lowest and highest prices of Avocado?
#2. What is the highest region of avocado production?
#3. What is the average avocado prices in each year?
#4. What is the average avocado volume in each year?
#13 attributes are divided as 
#1. Date - The date of the observation
#2. AveragePrice: the average price of a single avocado
#3. Total Volume: Total number of avocados sold
#4. Total Bags: Total number  o bags
#5. Small Bags: Total number of Small bags
#6. Large Bags: Total number of Large bags
#7. XLarge Bags: Total number of XLarge bags
#8. type: conventional or organic
#9. year: the year
#10. region: the city or region of the observation
#11. 4046: Total number of avocados with PLU 4046 sold
#12. 4225: Total number of avocados with PLU 4225 sold
#13. 4770: Total number of avocados with PLU 4770 sold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
df = pd.read_csv(r'C:\Users\Noman Mahmood\Desktop\DS\Projects\Avacado/avocado.csv')
df.info()
df.head()
df.isnull().sum()
df = df.drop(['Unnamed: 0','4046','4225','4770','Date'],axis=1)
df.head()
def get_avarage(df,column):
    """
    Description: This function to return the average value of the column 

    Arguments:
        df: the DataFrame. 
        column: the selected column. 
    Returns:
        column's average 
    """
    return sum(df[column])/len(df)
def get_avarge_between_two_columns(df,column1,column2):
    """
    Description: This function calculate the average between two columns in the dataset

    Arguments:
        df: the DataFrame. 
        column1:the first column. 
        column2:the scond column.
    Returns:
        Sorted data for relation between column1 and column2
    """
    
    List=list(df[column1].unique())
    average=[]

    for i in List:
        x=df[df[column1]==i]
        column1_average= get_avarage(x,column2)
        average.append(column1_average)

    df_column1_column2=pd.DataFrame({'column1':List,'column2':average})
    column1_column2_sorted_index=df_column1_column2.column2.sort_values(ascending=False).index.values
    column1_column2_sorted_data=df_column1_column2.reindex(column1_column2_sorted_index)
    
    return column1_column2_sorted_data

def plot(data,xlabel,ylabel):
    """
    Description: This function to draw a barplot

    Arguments:
        data: the DataFrame. 
        xlabel: the label of the first column. 
        ylabel: the label of the second column.
    Returns:
        None
    """
        
    plt.figure(figsize=(15,5))
    ax=sns.barplot(x=data.column1,y=data.column2,palette='rocket')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(('Avarage '+ylabel+' of Avocado According to '+xlabel));
    
    #lowest and Highest Price Region
    data1 = get_avarge_between_two_columns(df,'region','AveragePrice')
plot(data1,'Region','Price ($)')
print(data1['column1'].iloc[-1], " is the region producing avocado with the lowest price.")
#Highest Production of Product
data2 = get_avarge_between_two_columns(df,'region','Total Volume')
sns.boxplot(x=data2.column2).set_title("Figure: Boxplot repersenting outlier columns.")
outlier_region = data2[data2.column2>10000000]
print(outlier_region['column1'].iloc[-1], "is outlier value")
outlier_region.index
data2 = data2.drop(outlier_region.index,axis=0)
plot(data2,'Region','Volume')
# year base trend of Product
data3 = get_avarge_between_two_columns(df,'year','AveragePrice')
plot(data3,'year','Price')
data4 = get_avarge_between_two_columns(df,'year','Total Volume')
plot(data4,'year','Volume')
#model used linear regression 
df['region'] = df['region'].astype('category')
df['region'] = df['region'].cat.codes

df['type'] = df['type'].astype('category')
df['type'] = df['type'].cat.codes
df.info()
df.head()
# split data into X and y
X = df.drop(['AveragePrice'],axis=1)
y = df['AveragePrice']

# split data into traing and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=15)

print("training set:",X_train.shape,' - ',y_train.shape[0],' samples')
print("testing set:",X_test.shape,' - ',y_test.shape[0],' samples')

# bulid and fit the model
model = LinearRegression(normalize=True)
model.fit(X_train,y_train)

# prediction and calculate the accuracy for the testing dataset
test_pre = model.predict(X_test)
test_score = r2_score(y_test,test_pre)
print("The accuracy of testing dataset ",test_score*100)

# prediction and calculate the accuracy for the testing dataset
train_pre = model.predict(X_train)
train_score = r2_score(y_train,train_pre)
print("The accuracy of training dataset ",train_score*100)

#The model doesn't work well with this dataset, In order to the avocado prices were near together