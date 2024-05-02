import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
print(df)

print(df.columns)

x=df[['ID', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
       'tax', 'ptratio', 'black', 'lstat']]
print(x)
y=df['medv']
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
print(y_predict)

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

print(np.sqrt(mean_squared_error(y_test,y_predict)))

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_predict)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--')
plt.show()