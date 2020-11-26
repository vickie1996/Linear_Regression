import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_frame= pd.read_csv('E:\Projects\LR\Linear Regression CH\student_scores.csv')
data_frame

x = data_frame['Hours'].values.reshape(-1,1)
y = data_frame['Scores'].values.reshape(-1,1)

x_train, y_train = x[0:20], y[0:20]
x_test, y_test = x[20:], y[20:]

model = LinearRegression().fit(x_train,y_train)

y_predictions = model.predict(x_test)
y_predictions

plt.plot(x_test, y_predictions,'o')
plt.plot(x_test, y_test,'ro')
plt.show()

print(model.coef_)
print(model.intercept_)
