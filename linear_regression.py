# import libraries 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Load dataset 
data = pd.read_csv(r"weight-height.csv")
#print(data)
data.plot(kind= 'scatter', x= 'Height', y= 'Weight')
#plt.show()
#data.plot(kind ='box')
plt.show()
#print(data.corr()) #correlation coefficients 
#change to dataframe variables
Height = pd.DataFrame(data['Height'])
Weight = pd.DataFrame(data['Weight'])

#split data to train and test parts 
x_train, x_test, y_train, y_test = train_test_split(Height, Weight, test_size=0.33)


#build the model 
model= linear_model.LinearRegression()
#train 
model.fit(x_train, y_train)
print(model.coef_)

#calculate score 
score_train= model.score(x_train, y_train)
print("train score", score_train)

#predictions for testing the model 
predictions = model.predict(x_test)
print("predictions : ", predictions)
print("new height to predict " )
new_height = input()
height_predict =model.predict([[new_height]])
print("prediction of new input height: ",height_predict)
