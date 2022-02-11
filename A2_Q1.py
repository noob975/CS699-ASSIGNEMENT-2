# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:17:06 2022

@author: Aniruddha
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization,Conv2D,Conv2DTranspose,LeakyReLU,Activation,Flatten,Dense,Reshape,Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError
from tensorflow.keras.metrics import CosineSimilarity as CS
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

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



#%%

autoencoder = Sequential([
    Dense(64, activation = 'relu', input_shape = (200*300,)),
    Dense(200*300, activation = 'linear')])

opt = Adam(learning_rate=0.001)
autoencoder.compile(optimizer=opt,
              loss = MeanSquaredError(), #because images are of large size, they may be far apart in traditional MSE.
              metrics = CS())
#%%
history = autoencoder.fit(
    X_train, X_train,
    shuffle = True,
    epochs = 90,
    verbose = 1,
)
#%%

what = autoencoder.predict(X_train)
    
for i in range(0,40):

    this = what[i].reshape((300,200))
    comparison_plot(this, X_train[i].reshape((300,200)), index = i)

# this = what[16].reshape((300,200))
# comparison_plot(this, X_train[16].reshape((300,200)))
#%%
what = autoencoder.predict(X_test)
for i in range(0,20):

    this = what[i].reshape((300,200))
    comparison_plot(this, X_test[i].reshape((300,200)), index = i)




