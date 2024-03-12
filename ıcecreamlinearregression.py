import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as ipy


df=pd.read_csv("Ice Cream Sales - temperatures.csv")

df.head(5)

df[['Temperature','Ice Cream Profits']].describe().\
style.background_gradient(cmap=sns.light_palette('red', as_cmap=True))

plt.figure(figsize=(5,5))
sns.lineplot(data=df,x='Temperature',y='Ice Cream Profits')
plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(df.drop('Ice Cream Profits',axis=1),df['Ice Cream Profits'],test_size=0.3)

print('The training dataset has {} values and the test dataset {} values.'.format(X_train.shape[0], X_test.shape[0]))

from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x if x.ndim > 1 else x.reshape(-1, 1)
        self.y = y
        self.__correlation_coefficient = self.__correlation()
        self.__inclination = self.__inclination()
        self.__intercept = self.__intercept()

    def __correlation(self):
        corr = cov(self.x.T, self.y, bias=True)[0][1]
        var_x = var(self.x, ddof=1)
        var_y = var(self.y, ddof=1)

        return corr / sqrt(var_x * var_y)

    def __inclination(self):
        stdx = std(self.x, ddof=1)
        stdy = std(self.y, ddof=1)
        return self.__correlation_coefficient * (stdy / stdx)
     
    def __intercept(self):
        meanx = mean(self.x)
        meany = mean(self.y)
        return meany - self.__inclination * meanx

    def predict(self, value):
        return self.__intercept + (self.__inclination * value)
    
lr = LinearRegression(X_train,Y_train)


predict_value = []
for i in range(len(X_test)):
    predict_value.append(float(lr.predict(X_test.iloc[i]).round(2).item()))
    
predict_value = pd.DataFrame(predict_value, columns=['PredictedValue'])

Y_test.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

predict_value['Real_Value'] = Y_test
predict_value['Temperature'] = X_test

predict_value


plt.figure(figsize=(16,9))
plt.title("cream")

ax1 = sns.lineplot(data=predict_value, x='Temperature', y='Real_Value', color='blue', label='Real Value')

ax2 = plt.twinx()

sns.lineplot(data=predict_value, x='Temperature', y='PredictedValue', ax=ax2, color='green', label='Predicted Value')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()