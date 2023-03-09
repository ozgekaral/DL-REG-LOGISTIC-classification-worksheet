# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:53:48 2023

@author: user202
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from  keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pickle
import cv2
from tensorflow.keras.utils import to_categorical

path ='myData'
listm=os.listdir(path)
classes = len(listm)
print(classes)

classNo=[]
images=[]

for i in range(classes):
    imagelist=os.listdir(path + '\\'+str(i))
    for j in imagelist:
        img=cv2.imread(path + '\\'+str(i)+ '\\'+j)
        img=cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(1)
print(len(images))
print(len(classNo))

images=np.array(images)
classNo=np.array(classNo)

x_train, x_test, y_train, y_test =train_test_split(images, classNo, random_state=42, test_size=0.5)
x_train, x_validation, y_train, y_validation =train_test_split(images, classNo, random_state=42, test_size=0.5)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
 
#Preprocess

def preprocess(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255.0
    
    return img

#img=preprocess(x_train[20])
#img=cv2.resize(img,(250,250))
#cv2.imshow('preimg',img)

# map method
x_train=np.array(list(map(preprocess, x_train)))
x_test=np.array(list(map(preprocess, x_test)))
x_validation=np.array(list(map(preprocess, x_validation)))

x_train=x_train.reshape(-1,32,32,1)
x_test=x_test.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-1,32,32,1)

#Data generate

dataC=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, rotation_range=20)

dataC.fit(x_train)

#Like One-Hot Encode

y_train=to_categorical(y_train, classes)
y_test=to_categorical(y_test, classes)
y_validation=to_categorical(y_validation, classes)

#Model

model=Sequential()
model.add(Conv2D(filters=8, kernel_size=(5,5), input_shape=(32,32,1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2 )))

model.add(Dropout(0.2))
model.Flatten()
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(units=classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accurary'] )

batch_size=250

#Training
hist=model.fit_generator(dataC.flow(x_train, y_train, batch_size=batch_size),validation_data=(x_validation, y_validation),
                         epochs=15, step_per_epoch=x_train.shape[0]//batch_size, shuffle=1)

pickle_out=open('model_trained.p','wb')
pickle.dump(model, pickle_out)
pickle_out.close()

hist.history.keys()
plt.figure()
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='vali loss')
plt.legend()
plt.show()

hist.history.keys()
plt.figure()
plt.plot(hist.history['accurarcy'], label='train accurarcy')
plt.plot(hist.history['val_accurarcy'], label='accurarcy')
plt.legend()
plt.show()

score=model.evaluate(x_test, y_test, verbose=1)
print('test_loss:', score[0])
print('test accurarcy:', score[1])

y_pred=model.predict(x_validation)
y_pred_class=np.argmax(y_pred, axis=1)
y_true=np.argmax(y_validation, axis=1)

cm=confusion_matrix(y_true, y_pred_class)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True, linewidths=0.01, cmap='Greens', linecolor='gray', fmt='if', ax=ax)
plt.xlabel('predict')
plt.ylabel('true')
plt.show()




