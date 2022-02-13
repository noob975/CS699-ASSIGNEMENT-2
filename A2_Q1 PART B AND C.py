# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:17:06 2022

@author: Aniruddha
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation,Flatten,Dense,Reshape,Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError
from tensorflow.keras.metrics import CosineSimilarity as CS
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 40,
        }
#%%

img_path = r"C:\Users\Aniruddha\Desktop\CS699\A2\Group_8\train\butterfly\image_0002.jpg"

def prepare_image(img):
    img = cv2.resize(img, (200,300)) #cv2.resize takes breadth x length as resize inputs
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
bfly = prepare_image(plt.imread(img_path))

def normalize(img):
    new = (img - tf.reduce_min(img))/ (tf.reduce_max(img)- tf.reduce_min(img))
    return new

def comparison_plot(img1, img2,index,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(20,50))
    plt.subplot(1,2,1)
    plt.title(f"Reconstructed{index}", fontdict=font)
    plt.imshow(img1, cmap = "gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title(f"Original{index}", fontdict=font)
    plt.imshow(img2, cmap = "gray")
    plt.axis("off")
    return 0

train_folder_path = r"C:\Users\Aniruddha\Desktop\CS699\A2\Group_8\train"
test_folder_path = r"C:\Users\Aniruddha\Desktop\CS699\A2\Group_8\test"

def read_images(path):
    
    output = {}
    os.chdir(path)
    for classes in os.listdir():
        print(classes)
        os.chdir(path+f"\\{classes}")
        images_of_one_class = []
        for img in os.listdir():
            I = plt.imread(img)
            
            try: #skips grayscale conversion for images that are already grayscale
                I = prepare_image(I)
            except:
                I = cv2.resize(I, (200,300))
            
            images_of_one_class.append(I)
        output[classes] = images_of_one_class
        
    return output

train_images = read_images(train_folder_path) 
test_images = read_images(test_folder_path)

#%%

X_train = np.zeros((0,60000))

for i in train_images:
    c = np.array(train_images[i])
    c = c.reshape(-1, 200*300)
    X_train = np.concatenate((X_train, c))

X_test = np.zeros((0,60000))

for i in train_images:
    c = np.array(test_images[i])
    c = c.reshape(-1, 200*300)
    X_test = np.concatenate((X_test, c))

c1 = np.zeros((40))
c2 = np.ones((40))
c3 = c2+1
y_train = np.concatenate((c1,c2,c3))

c1 = np.zeros((20))
c2 = np.ones((20))
c3 = c2+1
y_test = np.concatenate((c1,c2,c3))
#%%
'''
Three layer accuracy
'''
def threeLayer(n1, n2): #n1 and n2 are the encoding number of neurons for 3-layer-autoencoder

    autoencoder = Sequential([
        Dense(n1, activation = 'relu', input_shape = (200*300,)),
        Dense(n2, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(256, activation = 'relu'),
        Dense(3, activation = 'softmax')])
    
    opt = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt,
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    history = autoencoder.fit(
        X_train, y_train,
        shuffle = True,
        epochs = 30, #after this loss starts oscillating
        verbose = 1,
    )
    
    what = autoencoder.predict(X_test)
    y_pred = []
    for i in range(len(what)):
        c = np.argmax(what[i])
        y_pred.append(c)
    
    return y_pred

actual1 = threeLayer(1024, 256)
actual2 = threeLayer(256, 32)
actual3 = threeLayer(128, 64)
#%%
print("\n<------- Accuracy Score ------->")

print("1024_256_1024:\n", accuracy_score(y_test, actual1))
print("256_32_256:\n", accuracy_score(y_test, actual2))
print("128_64_128:\n", accuracy_score(y_test, actual3))

print("\n<------- Confusion Matrix ------->")

print("\n1024_256_1024:\n", confusion_matrix(y_test, actual1))
print("\n256_32_256:\n", confusion_matrix(y_test, actual2))
print("\n128_64_128:\n", confusion_matrix(y_test, actual3))

#%%
'''
one layer accuracy
'''

def oneLayer(n1): #n1 is the the encoding number of neurons for 1-layer-autoencoder

    autoencoder = Sequential([
        Dense(n1, activation = 'relu', input_shape = (200*300,)),
        Dense(128, activation = 'relu'),
        Dense(256, activation = 'relu'),
        Dense(3, activation = 'softmax')])
    
    opt = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt,
                  loss = 'sparse_categorical_crossentropy', 
                  metrics = ['accuracy'])
    
    history = autoencoder.fit(
        X_train, y_train,
        shuffle = True,
        epochs = 30,
        verbose = 1,
    )
    
    what = autoencoder.predict(X_test)
    y_pred = []
    for i in range(len(what)):
        c = np.argmax(what[i])
        y_pred.append(c)
        
    fig = plt.figure(figsize=(10,10))  
    maximal_weights =  np.array(autoencoder.weights[0])
    for i in range(2700):
        maximal_weights[:,0] /= np.linalg.norm(maximal_weights[:,0])
    for i in range(1,17):
        fig.add_subplot(8, 8, i)
        plt.imshow(maximal_weights[:,i-1].reshape(300,200),cmap='gray')
        plt.axis('off')
    
    return y_pred
#%%
actual1 = oneLayer(64)
actual2 = oneLayer(128)
actual3 = oneLayer(2700)



#%%



