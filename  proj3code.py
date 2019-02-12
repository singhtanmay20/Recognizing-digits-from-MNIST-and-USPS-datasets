#!/usr/bin/env python
# coding: utf-8

# ## Load MNIST on Python 3.x

# In[9]:


import pickle
import gzip
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
import numpy as np
from PIL import Image
import os
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[10]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f)
print(len(training_data),len(validation_data),len(test_data))
f.close()
print(test_data[1].shape)
list1=[]
list2=[]
list3=[]
list4=[]


# In[11]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPS/USPSdata/USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
            
USPSMat=np.array(USPSMat)
USPSTar=np.array(USPSTar)


# In[12]:


print(test_data[1].shape)
print(USPSMat.shape)
print(USPSTar.shape)


# In[69]:


input_size = 784
drop_out = 20
first_dense_layer_nodes  = 128
second_dense_layer_nodes  = 128
third_dense_layer_nodes = 10

def get_model():
        model = Sequential()
        model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(second_dense_layer_nodes, input_dim=input_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(third_dense_layer_nodes))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


# In[70]:


#def myDNN(training_data,validation_data,test_data):
    
model=get_model()

validation_data_split = 0.1
num_epochs =1000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Process Dataset
processedData=np.concatenate((training_data[0],validation_data[0]), axis=0);
processedLabel=np.concatenate((training_data[1],validation_data[1]), axis=0);
print(len(processedLabel))
processedLabel=keras.utils.to_categorical(processedLabel)

history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                    
                   )
    
    


# In[71]:


wrong   = 0.0
right   = 0.0
processedTestData  = test_data[0]
processedTestLabel = keras.utils.to_categorical(test_data[1])
lossMNIST,accuracyMNIST = model.evaluate(processedTestData, processedTestLabel, verbose=False) 
print("-----------------------------------------------------------------------------------------")
print("Accuracy from MNIST Neural Network dataset : "+ str(accuracyMNIST))
print("loss from MNIST Neural Network dataset : "+str(lossMNIST))
)




df1 = pd.DataFrame()

df2 = pd.DataFrame()
temp3=[accuracyMNIST]
list3.extend(temp3)
temp4=[num_epochs]
list4.extend(temp4)
df2["number of epoch"]=list4
df2["acc"]=list3
df2.plot(grid=True)
print(df2)
plt.savefig('MNISTNN_accuracy_noofepoch.png', bbox_inches='tight')
plt.show()


df = pd.DataFrame(history.history)
df.plot(grid=True)
plt.savefig('MNISTNN_50_epoch.png', bbox_inches='tight')
plt.show()


# In[72]:


wrong   = 0.0
right   = 0.0
processedTestData  = USPSMat
processedTestLabel = keras.utils.to_categorical(USPSTar)
lossUSPS,accuracyUSPS = model.evaluate(processedTestData, processedTestLabel, verbose=False) 
print("-----------------------------------------------------------------------------------------")
print("Accuracy from USPS Neural Network dataset : "+ str(accuracyUSPS))
print("loss from USPS Neural Network dataset : "+str(lossUSPS))

df1 = pd.DataFrame()
temp1=[accuracyUSPS]
list1.extend(temp1)
temp2=[num_epochs]
list2.extend(temp2)
df1["number of epoch"]=list2
df1["acc"]=list1
df1.plot(grid=True)
print(df1)
plt.savefig('USPS_accuracy_noofepoch.png', bbox_inches='tight')
plt.show()


# In[85]:


predicteduspsnn1=model.predict(processedTestData)
predicteduspsnn=np.argmax(predicteduspsnn1, axis=1)
print(predicteduspsnn)

predictedmnist1=model.predict(test_data[0])
predictedmnist=np.argmax(predictedmnist1, axis=1)

print(USPSTar.shape)

CMTest=sklearn.metrics.confusion_matrix(USPSTar,predicteduspsnn)
sumtest=sum(CMTest.diagonal())
sumtest=sumtest/float(len(USPSTar))
print(CMTest)
print("Testing accuracy usps",sumtest)

print(test_data[1].shape)
CMTestmnist=sklearn.metrics.confusion_matrix(test_data[1],predictedmnist)
sumtestmnist=sum(CMTestmnist.diagonal())
sumtestmnist=sumtestmnist/float(len(test_data[0]))
print(CMTestmnist)
print("Testing accuracy mnist",sumtestmnist)


# # Logistic Regression

# In[89]:


#def softmax(inputs):
 #   return np.exp(inputs) / np.array(sum(np.exp(inputs)))

def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear))
    norms = sum(exp)
    return exp / norms

def predict(features, weights):
    final=[]
    z = np.dot(features, weights)
    for i in range(len(z)):
        final.append(softmax(z[i]))
    return final

def loss_function(predictions,labels):#used to calculate the error of the model
    #The cost should continuously decrease for the model to work correctly 
    class1_cost=0   
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            class1_cost=class1_cost+(-labels[i,j]*np.log(predictions[i,j]))
    return class1_cost



def gradient(trainingFeatures,trainingLabels,predictionsTraining):
    predictionsTraining=np.array(predictionsTraining)
    return np.dot(trainingFeatures.T,predictionsTraining-trainingLabels)
    


# In[95]:


#def logisticRegression(trainingData,validationData,testData,name):
    
lr=0.01
mini_Batch_Size=100


trainingData=np.transpose(training_data[0])
testData=np.transpose(test_data[0])
validationData=np.transpose(validation_data[0])

weights=np.random.randn(784,10)#randomly initializing the weights having dimension equal to (num of features,target vector)
#print(weights.shape)


trainingFeatures=training_data[0]
trainingLabels=keras.utils.to_categorical(training_data[1])
testingFeatures=test_data[0]
testingLabels=keras.utils.to_categorical(test_data[1])
validationFeatures=validation_data[0]
validationLabels=keras.utils.to_categorical(validation_data[1])

no_Of_Batches=len(trainingFeatures)/mini_Batch_Size
trainingProb=[]
validationProb=[]
testingProb=[]

loss1=-99999999
loss=0
for k in range(400):
    j=0
    predictionsTraining=[]
    predictionsValidation=[]
    predictionsTest=[]
    predictionsUSPS=[]
    
    if (loss1==loss):
        break
    loss1=loss
    for i in range(no_Of_Batches):
        
        
        trainingFeatures1=trainingFeatures[j:j+mini_Batch_Size,:]
        trainingLabels1=trainingLabels[j:j+mini_Batch_Size,:]
        
        validationFeatures1=validationFeatures[j:j+mini_Batch_Size,:]
        validationLabels1=validationLabels[j:j+mini_Batch_Size,:]
        
        testFeatures1=testingFeatures[j:j+mini_Batch_Size,:]
        testLabels1=testingLabels[j:j+mini_Batch_Size,:]
        
        USPSFeatures1=USPSMat[j:j+mini_Batch_Size,:]
       # USPSLabels1=USPSTar[j:j+mini_Batch_Size,:]
        
        j=j+mini_Batch_Size
        
        
        N = len(trainingFeatures1)
        predictionsTraining1=np.array(predict(trainingFeatures1, weights))
        loss=loss_function(predictionsTraining1,trainingLabels1)#calculating the error in prediction    

        gradient1=np.array(gradient(trainingFeatures1,trainingLabels1,predictionsTraining1))
        weight_Next=weights-(1)*lr*gradient1
        weights=weight_Next
        
        predictionsValidation1=np.array(predict(validationFeatures1, weights))
        predictionsTest1=np.array(predict(testFeatures1, weights))
        
        predictionsUSPS1=np.array(predict(USPSFeatures1, weights))
        
        predictionsValidation.extend(predictionsValidation1)
        predictionsTraining.extend(predictionsTraining1)
        predictionsTest.extend(predictionsTest1)
        predictionsUSPS.extend(predictionsUSPS1)
    
    
        
    
    
    


# In[97]:


predictionsTraining=np.asarray(predictionsTraining)
CMTr=sklearn.metrics.confusion_matrix(trainingLabels.argmax(axis=1),predictionsTraining.argmax(axis=1))
sumtr=sum(CMTr.diagonal())
sumtr=sumtr/float(len(trainingLabels))
print(CMTr)
print("Training Accuracy MNIST",sumtr)

predictionsValidation=np.asarray(predictionsValidation)
CMVa=sklearn.metrics.confusion_matrix(validationLabels.argmax(axis=1),predictionsValidation.argmax(axis=1))
print(CMVa)
sumva=sum(CMVa.diagonal())
sumva=sumva/float(len(validationLabels))
print("Validation Accuracy MNIST",sumva)

predictionsTest=np.asarray(predictionsTest)
CMTe=sklearn.metrics.confusion_matrix(testingLabels.argmax(axis=1),predictionsTest.argmax(axis=1))
print(CMTe)
sumte=sum(CMTe.diagonal())
print("Testing Accuracy MNIST",sumte/float(len(testingLabels)))


USPSTar1=keras.utils.to_categorical(USPSTar)
print(USPSTar.shape)
predictionsUSPS=np.asarray(predictionsUSPS)
CMUSPS=sklearn.metrics.confusion_matrix(USPSTar1.argmax(axis=1),predictionsUSPS.argmax(axis=1))
print(CMUSPS)
sumusps=sum(CMUSPS.diagonal())
sumusps=sumusps/float(len(predictionsUSPS))
print("Accuracy USPS",sumusps)


df3=pd.DataFrame()
list8=[lr]
list5=[sumtr]
list6=[sumva]
list7=[sumusps]
list9=[sumte]

df3["learning rate"]=list8
df3["mnist training acc log"]=list5
df3["mnist validation acc log"]=list6
df3["mnist usps acc log"]=list7
df3["mnist testing acc log"]=list9

with open('logistic.csv', 'a') as f:
    df3.to_csv(f)


# In[98]:


logisticUSPSPred= np.argmax(predictionsUSPS, axis=1)
print(logisticUSPSPred)


# # Random Forest

# In[172]:


from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):#function to get the mnist-original.mat file
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


# In[217]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from pylab import *

fetch_mnist()
mnist = fetch_mldata("/home/tanmay/scikit_learn_data/mldata/mnist-original")
#mnist_data = mnist["data"].T
#mnist_target = mnist["label"][0]
#mnist = fetch_mldata('~/scikit_learn_data/mldata/mnist-original.mat')
n_train = 50000
n_test = 10001
n_estimators=300

indices = arange(len(mnist_data))
train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)
X_train, y_train = mnist_data[train_idx], mnist_target[train_idx]
X_test, y_test = mnist_data[test_idx], mnist_target[test_idx]
classifier2 = RandomForestClassifier(n_estimators=100);# uses scikit random fores function 
classifier2.fit(X_train, y_train)


# In[219]:


MnistTrAccRF=classifier2.score(X_train,y_train)
MnistTeAccRF=classifier2.score(X_test,y_test)
print ('RF accuracy: TRAINING', MnistTrAccRF)
print ('RF accuracy: TESTING',MnistTeAccRF )


print(mnist_target.shape)
MnistAccRF=classifier2.predict(mnist_data[train_idx])
MnistAcctestRF=classifier2.predict(X_test)

print(MnistAcctestRF.shape)

USPSRFPred=classifier2.predict(USPSMat)
print(USPSRFPred)
print(USPSTar)
count=0
for i in range(len(USPSRFPred)):
    if(USPSRFPred[i]==USPSTar[i]):
        count=count+1
print(count)
print(len(USPSRFPred))
USPSRFAcc=float(count)/len(USPSRFPred)
print('RF USPS accuracy: '+str(USPSRFAcc))


# In[201]:


CMTestmnist=sklearn.metrics.confusion_matrix(mnist_target[train_idx],MnistAccRF)
summnist=sum(CMTestusps.diagonal())
summnist=sumusps/float(len(test_data[1]))
print(CMTestmnist)
print("Testing Accuracy MNIST",summnist)



CMTestusps=sklearn.metrics.confusion_matrix(USPSTar,USPSRFPred)
sumusps=sum(CMTestusps.diagonal())
sumusps=sumusps/float(len(USPSTar))
print(CMTestusps)
print("Testing Accuracy usps",sumusps)


# In[202]:



df4=pd.DataFrame()
list9=[n_estimators]
list10=[USPSRFAcc]
list11=[MnistTrAccRF]
list12=[MnistTeAccRF]

df4["no of tress"]=list9
df4["accuracy usps"]=list10
df4["mnist train"]=list11
df4["mnist test"]=list12
with open('RandomForest.csv', 'a') as f1:
    df4.to_csv(f1)


# # SVM

# In[146]:


classifier1 = SVC(kernel='rbf', C=2, gammma = 0.05);
classifier1.fit(X_train, y_train) 

MnistAccSVM=classifier1.predict(mnist_data[train_idx])
MnistAcctestSVM=classifier1.predict(X_test)

USPSSVMPred=classifier1.predict(USPSMat)
count=0
for i in range(len(USPSSVMPred)):
    if(USPSSVMPred[i]==USPSTar[i]):
        count=count+1
print(count)
print(len(USPSSVMPred))
USPSSVMAcc=float(count)/len(USPSSVMPred)
print('SVM USPS accuracy: '+str(USPSSVMAcc))


# # ensemble

# In[225]:


USPSRFPred=USPSRFPred.astype(int)
print(predicteduspsnn)
print(logisticUSPSPred)
print(USPSRFPred)
temp=np.vstack((predicteduspsnn, logisticUSPSPred))
print(temp.shape)
temp1=np.vstack((temp,USPSRFPred))
final_Ensemble=temp1.T
print(final_Ensemble)
a=[]
for i in range(len(final_Ensemble)):
    a.append(max(final_Ensemble[i],key=final_Ensemble[i].tolist().count))
a=np.array(a)

count=0
for i in range(len(a)):
    if(a[i]==USPSTar[i]):
        count=count+1
print(count)
print(len(a))
Ensemble_Acc=float(count)/len(a)
print('ensembled USPS accuracy: '+str(Ensemble_Acc))




CM1=sklearn.metrics.confusion_matrix(USPSTar,a)
summnist=sum(CM1.diagonal())
summnist=sumusps/float(len(USPSTar))
print(CM1)


# In[230]:


predictedmnist
predictionsTest
MnistTeAccRF
temp=np.vstack((predictedmnist, predictionsTest.argmax(axis=1)))
print(MnistAcctestRF.shape)
temp1=np.vstack((temp,MnistAcctestRF))
final_Ensemble=temp1.T
print(final_Ensemble)
b=[]
for i in range(len(final_Ensemble)):
    b.append(max(final_Ensemble[i],key=final_Ensemble[i].tolist().count))#appending the maximum occurence variable in b
b=np.array(b)

count=0
test_data2=test_data[1]
for i in range(len(b)):
    if(b[i]==test_data2[i]):
        count=count+1
print(count)
print(len(b))
Ensemble_Acc=float(count)/len(b)
print('ensembled mnist accuracy: '+str(Ensemble_Acc))



CM2=sklearn.metrics.confusion_matrix(test_data[1],b)##calculating confusion matrix
summnist=sum(CM2.diagonal())
summnist=sumusps/float(len(USPSTar))
print(CM2)


# In[ ]:





# In[ ]:




