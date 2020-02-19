# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:27:59 2020

@author: Rohan Dixit
"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('musk_csv.csv')

#Set the value of X and y with iloc function 
X=dataset.iloc[:, 3:169].values
y=dataset.iloc[:, 169].values

# Splitting the dataset into the Training set and Test set in 80:20 ratio
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#PRE PROCEESING OF DATASET
# Feature  Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu', input_dim = 166))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adadelta' ,loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 16, nb_epoch = 50,validation_data=(X_test,y_test),shuffle=True,verbose=1)

#save the model with .h5 extension
classifier.save('musk_model.h5')

#Plot the Graph of model accuracy 

import matplotlib.pyplot as plt 
plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Plot the Graph of model loss

plt.plot(classifier.history.history['loss'])
plt.plot(classifier.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Load the model
from tensorflow.keras.models import load_model
classifier = load_model('musk_model.h5')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# for true or false result 
y_pred = (y_pred > 0.5)


# find and make  the accuracy score, classification report and confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("Classification Report is :\n",classification_report(y_test,y_pred))
print("Accuracy Score is :",accuracy_score(y_test,y_pred))
print("Confusion matrix is :\n",confusion_matrix(y_test, y_pred))

