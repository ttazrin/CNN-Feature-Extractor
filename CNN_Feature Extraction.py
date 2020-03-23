# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:53:18 2020

@author: ttazrin
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
#from PIL import Image
from sklearn.model_selection import train_test_split
from numpy import newaxis
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


#method that returns CNN extracted features
def featureExtraction(x,y):
    x = x.to_numpy()
    y = y.to_numpy()
    x = x[:, :, newaxis]
    
    verbose, epochs, batch_size = 1, 50, 1
    n_timesteps, n_features = x.shape[1], x.shape[2]
    model = Sequential()
    #model.build(1344)
    model.add(Conv1D(filters=100, kernel_size=6, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=1))
    model.add(BatchNormalization())
    
    model.add(Conv1D(filters=75, kernel_size=4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=1))
    model.add(BatchNormalization())
    
    model.add(Conv1D(filters=50, kernel_size=2, activation='relu'))
    
    model.add(Conv1D(filters=12, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(15, activation='relu', name ='feature_dense'))
    model.add(Dense(1, activation='softmax'))
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #fit network
    model.fit(x, y, validation_split=0.02, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.summary()
    
    feature_layer = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_dense').output)
    feature_layer.summary()

    new_data = feature_layer.predict(x)
    new_data = pd.DataFrame(new_data )
    return new_data



#START
data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
      
#calling feature extractor function  
x_train_features = featureExtraction(X_train,y_train)
print('x_train_features shape:', x_train_features.shape)
x_train_features.head(5)
   
clf = RandomForestClassifier(n_estimators = 500, random_state = 42
                                 , max_features = 'log2')
clf.fit(x_train_features,y_train)

x_test_features = featureExtraction(X_test,y_test)

y_pred = clf.predict(x_test_features)
accuracy = accuracy_score(y_test, y_pred)
            
 
    
    
