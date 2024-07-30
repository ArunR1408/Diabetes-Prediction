import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

d = pd.read_csv('diabetes.csv')
print(d)
print('\n')
print(d.describe())
print('\n')
print(d['Outcome'].value_counts())

# 0 = Non-diabetic
# 1 = Diabetic

x = d.drop(columns='Outcome',axis=1)
y = d['Outcome']
print('\n')
print(x)

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print('\n')
print(standardized_data)

x = standardized_data

print('\n')
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print('\n')
print(x_train.shape)
print('\n')
print(x_test.shape)

# Train the model

clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)

x_train_prediction = clf.predict(x_train)
print('\n') #Accuracy Score
print(accuracy_score(x_train_prediction,y_train))

# Accuracy on test data
x_test_prediction = clf.predict(x_test)
print('\n') #Accuracy Score
print(accuracy_score(x_test_prediction,y_test))

input_sample = (5,166,72,19,175,22.7,0.6,51)
input_np_array = np.asarray(input_sample)
input_np_array_reshaped = input_np_array.reshape(1,-1)
std_data = scaler.transform(input_np_array_reshaped)
print('\n')
print(std_data)

prediction = clf.predict(std_data)
print('\n')
print(prediction)

if (prediction[0] == 0 ):
    print("Person is not diabetic")
else:
    print("Person is diabetic")
