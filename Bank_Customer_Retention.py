#import library

import numpy as np   #contain maths
import tensorflow as tf
import pandas as pd #to import dataset and to manage data set

#Initializing the ANN#
ann = tf.keras.models.Sequential() #Intializes ANN as a sequence of layers


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()

def ANN():
    #####Data Pre-Processing######

    #Importing Dataset#
    dataset=pd.read_csv('Churn_Modelling.csv') #load data set
    X=dataset.iloc[:,3:-1].values    #independnt variables
    y=dataset.iloc[:, -1].values     #dependent data**************MISTAKE

    #Encoding Categorical Data#
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    labelencoder_X=LabelEncoder()
    X[:,2]=labelencoder_X.fit_transform(X[:,2])
    #here problem is that machine learning algo thinks that 0<2 meaning
    # France is less than spain but this is not the case at all
    #hence we use dummy column buidling three column
    #meanig put 1 if that France is there for ex. and put 0 if not.
    ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
    X=np.array(ct.fit_transform(X))


    #Split Data in train and test set#
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    #Feature Scalling#It is very very important and compulsory when doing Deep Learning

    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)

    #####Building the ANN######


    #Adding input and first hidden layer#
    ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

    #Adding second hidden layer#
    ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

    #Adding the output Layer#
    ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
    #when two or more output, actication fucntion would be #softmax


    #####Training the ANN#####

    #Compiling the ANN#
    ann.compile(optimizer= 'adam',loss= 'binary_crossentropy',metrics=['accuracy'])
    # when more than two output use #categorical_crossentropy

    #Training the ANN on training set
    ann.fit(X_train,y_train,batch_size=32,epochs=100)

    #####Making the Predictions and evaluating the model#####


    #####Predicting The Test Result#####
    y_pred=ann.predict(X_test)
    y_pred=(y_pred > 0.5)# True->1(0.5 to 1)---> High chance to leave bank  False->0(0 to 0.5)---> Low chance to leave bank
    #print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

    #####Confusion Metrics#####
#     from sklearn.metrics import confusion_matrix, accuracy_score
#     cm=confusion_matrix(y_test,y_pred)
#     print(cm)
#     print(accuracy_score(y_test,y_pred))

#     from mlxtend.plotting import plot_confusion_matrix
#     import matplotlib.pyplot as plt


#     fig, ax = plot_confusion_matrix(conf_mat=cm)
#     plt.show()

def single_test():
    # Predicting the result of a single observation
    print(ann.predict(sc_X.transform([[1, 0, 0, 500, 1, 40, 3, 100000, 2, 1, 1, 70000]])) > 0.5)
    print(ann.predict(sc_X.transform([[1, 0, 0, 500, 1, 40, 3, 100000, 2, 1, 1, 70000]])))
    # nned to input info in 2d double bracket array
    # replace the categorical value as dummy value which map to one hot encoding values
    # then predict this values on scalled value varible using transform method cause our maodel is trained on scalled values during
    # feature scalling

if __name__ == '__main__':
    ANN()
    single_test()
