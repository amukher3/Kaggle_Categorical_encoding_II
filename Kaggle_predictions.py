# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:53:06 2020

@author: abhi0
"""

#Categorical encoding features II
#Kaggle challenge
#Open competition

import pandas as pd
dfTrain=pd.read_csv("C:/Users/abhi0/OneDrive/Documents/Kaggle_CategoricalEncoding_II/train.csv")
dfTest=pd.read_csv("C:/Users/abhi0/OneDrive/Documents/Kaggle_CategoricalEncoding_II/test_Kaggle.csv")


#Replacing the NaN's with the most frequent values:

#For binary variables:
for i in range(5):
    dfTest['bin_'+str(i)]=dfTest['bin_'+str(i)].fillna(dfTest['bin_'+str(i)].value_counts().idxmax())
    dfTrain['bin_'+str(i)]=dfTrain['bin_'+str(i)].fillna(dfTrain['bin_'+str(i)].value_counts().idxmax())
    
#For nominal variables:
for i in range(10):
    dfTest['nom_'+str(i)]=dfTest['nom_'+str(i)].fillna(dfTest['nom_'+str(i)].value_counts().idxmax())
    dfTrain['nom_'+str(i)]=dfTrain['nom_'+str(i)].fillna(dfTrain['nom_'+str(i)].value_counts().idxmax())
    
#For Ordinal Variables:    
for i in range(6):
    dfTest['ord_'+str(i)]=dfTest['ord_'+str(i)].fillna(dfTest['ord_'+str(i)].value_counts().idxmax())    
    dfTrain['ord_'+str(i)]=dfTrain['ord_'+str(i)].fillna(dfTrain['ord_'+str(i)].value_counts().idxmax())
    
#For cyclic-variables:Day
dfTest['day']=dfTest['day'].fillna(dfTest['day'].value_counts().idxmax())
dfTrain['day']=dfTrain['day'].fillna(dfTrain['day'].value_counts().idxmax())
       
##Dropping Day variable:
#dfTrain.drop(['day'],axis=1,inplace=True)
#dfTest.drop(['day'],axis=1,inplace=True) 
   
 
#For month-variables:Month
dfTest['month']=dfTest['month'].fillna(dfTest['month'].value_counts().idxmax())      
dfTrain['month']=dfTrain['month'].fillna(dfTrain['month'].value_counts().idxmax())

##Initial approach: Drop NaN's
#dfTrain=dfTrain.dropna()
#dfTest=dfTest.dropna()

################ Feature pre-processing: #####################

#dropping the ID column
dfTrain.drop(['id'],axis=1,inplace=True)
dfTest.drop(['id'],axis=1,inplace=True)


#dropping some of the nominal variables:
#nom_5,nom_6,nom_7,nom_8,nom_9
dfTrain.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)
dfTest.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)


########## Label encoding the binary variables ###############

# For 'bin_3' variable
from sklearn.preprocessing import LabelEncoder
labelencoder_bin_3 = LabelEncoder()
dfTrain['bin_3'] = labelencoder_bin_3.fit_transform(dfTrain['bin_3'])
dfTest['bin_3'] = labelencoder_bin_3.fit_transform(dfTest['bin_3'])

# For 'bin_4' variable
from sklearn.preprocessing import LabelEncoder
labelencoder_bin_4 = LabelEncoder()
dfTrain['bin_4'] = labelencoder_bin_4.fit_transform(dfTrain['bin_4'])
dfTest['bin_4'] = labelencoder_bin_4.fit_transform(dfTest['bin_4'])

########## Label encoding the ordinal variables ###############

for i in range(1,6):
   labelencoder_ord_i = LabelEncoder()
   dfTrain['ord_'+str(i)] = labelencoder_ord_i.fit_transform(dfTrain['ord_'+str(i)]) 
   dfTest['ord_'+str(i)] = labelencoder_ord_i.fit_transform(dfTest['ord_'+str(i)]) 


######### One-hot encoding for nominal variables ###########

for i in range(0,5):
    df_dummies=pd.get_dummies(dfTrain['nom_'+str(i)],prefix='nom_'+str(i)+'_category:')
    dfTrain=pd.concat([dfTrain,df_dummies],axis=1)
    dfTrain=dfTrain.drop(['nom_'+str(i)],axis=1)
    df_dummies=pd.get_dummies(dfTest['nom_'+str(i)],prefix='nom_'+str(i)+'_category:')
    dfTest=pd.concat([dfTest,df_dummies],axis=1)
    dfTest=dfTest.drop(['nom_'+str(i)],axis=1)


#For 'day' variable: 
df_dummies=pd.get_dummies(dfTrain['day'],prefix='day_category:')
dfTrain=pd.concat([dfTrain,df_dummies],axis=1)
dfTrain=dfTrain.drop(['day'],axis=1)
df_dummies=pd.get_dummies(dfTest['day'],prefix='day_category:')
dfTest=pd.concat([dfTest,df_dummies],axis=1)
dfTest=dfTest.drop(['day'],axis=1)

#For 'month' variable:
df_dummies=pd.get_dummies(dfTrain['month'],prefix='month_category:')
dfTrain=pd.concat([dfTrain,df_dummies],axis=1)
dfTrain=dfTrain.drop(['month'],axis=1)
df_dummies=pd.get_dummies(dfTest['month'],prefix='month_category:')
dfTest=pd.concat([dfTest,df_dummies],axis=1)
dfTest=dfTest.drop(['month'],axis=1)

####### Keeping a separate set just for tesing purpose ###### 
#dfTest=df[:round(len(df)*0.30)]
#dfTrain=df
#dfTrain=df

############### Separating the traning set into train and dev sets ##################
    
#For the training data frame separating into dependent and independednt variables. 
#Further,separating the dependednt variable into into training and dev.set
#25-75 ratio adapted. 
    
##Separating the independent variable:
Y=dfTrain['target']
#
X=dfTrain.drop(['target'],axis=1)
    
##### Splitting the datasets
##Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size = 0.30, random_state = 0) 

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import Dropout
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


################################# training the classifier ##################################
    
#Parameters arrived for
#grid search.
Activation=['sigmoid']
#Optimizer=['adam','adagrad','rmsprop','sgd']
Optimizer=['adam']
#BatchSize=[40,80,100,120,140,160,200,250,300]
BatchSize=[2000]
#with 250 batch size result already saved
#WITH 300 ~ it is 0.81 
#ActivityRegularizer=[0.1,0.5,0.01,0.05,0.001,0.005,0.0001,1e-20]
ActivityRegularizer=[1e-30]
auc_dev_set=[]
auc_train_set=[]   
iVal=[]
jVal=[]
kVal=[]
opVal=[] 
for i in BatchSize:
   for  j in Activation:
       for k in ActivityRegularizer:
           for op in Optimizer:
               
               # Initialising the ANN
               classifier = Sequential()
               
               # Adding the input layer or the first hidden layer
               classifier.add(Dense(units=X_train.shape[1],
                                   kernel_initializer =keras.initializers.he_normal(seed=None),
                                   activation = 'relu',
                                   input_dim = X_train.shape[1],
                                   activity_regularizer=l2(k)))
              
               for i in range(10):
                    # Adding the rest of the layers
                    classifier.add(Dense(units=round(X_train.shape[1]/2),
                                   kernel_initializer =keras.initializers.he_normal(seed=None),
                                   activation = 'relu',
                                   input_dim = X_train.shape[1],
                                   activity_regularizer=l2(k)))
                    #classifier.add(BatchNormalization())
                
               
               # Adding the output layer
               classifier.add(Dense(units = 1,kernel_initializer=keras.initializers.he_normal(seed=None),
                                    activation =j,
                                    activity_regularizer=l2(k)))
             
               
               # Compiling the ANN
               classifier.compile(optimizer = op, loss = 'binary_crossentropy', metrics = ['accuracy'])
               
               ##Checkpoint:
               #EarlyStopping(monitor='val_loss',verbose=1)
               checkpoint=ModelCheckpoint(filepath='C:/Users/Abhishek Mukherjee/Documents/kaggle_categoricalencoding_ii/best_model.h5', 
                                          monitor='val_loss', save_best_only=True)

               # Fitting the ANN to the Training set
               history=classifier.fit(X_train, y_train,validation_data =(X_dev,y_dev),
                              batch_size = i, 
                              epochs = 20,
                              callbacks=[checkpoint])
               
               
               
#Printing history:
print(history.history.keys())

#Accuracy history:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()
               
               
              
               
#               ######################### checking for the variance trade-off ###############################
#               
#               #Predictions on the train-set
#               y_pred_train=classifier.predict(X_train)
#               
#               #Predictions on the dev-set
#               y_pred_dev=classifier.predict(X_dev)
#               
#               #AUC on the train-set
#               from sklearn.metrics import roc_curve, auc
#               fpr, tpr, _ = roc_curve(y_dev, y_pred_dev)
#               auc_train_set.append(auc(fpr, tpr))
#               print(auc_train_set)
#               
#               iVal.append(i)
#               jVal.append(j)
#               kVal.append(k)
#               opVal.append(op)
               
               
#               ##################### Making predictions on unseen data ######################
#
#               # Predicting the Test set results
#               y_pred_test = classifier.predict(dfTest)
#               

#               #AUC on the dev-set
#               from sklearn.metrics import roc_curve, auc
#               fpr, tpr, _ = roc_curve(y_dev, y_pred_dev)
#               auc_dev_set.append(auc(fpr, tpr))
#               print(auc_dev_set)


