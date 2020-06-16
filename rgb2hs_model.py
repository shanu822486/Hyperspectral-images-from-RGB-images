
import tensorflow
import keras as k
from tensorflow.keras.layers import Conv2D,Concatenate,Input,Convolution2D,concatenate
from tensorflow.keras.losses import mean_absolute_error
import numpy as np
import os, stat, sys
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.optimizers import Adam
import  math
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
import tensorflow as tf
from skimage import io, img_as_uint, img_as_ubyte
from scipy.io import  loadmat
from tensorflow.keras.layers import *




def build_net(image):
  ip=Input((image,image,3))
  start_neuron=8
  x=Conv2D(start_neuron*1,(3,3),activation='relu',padding='same')(ip)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*2,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*4,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*8,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*16,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*32,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*64,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)
  x=Conv2D(start_neuron*128,(3,3),activation='relu',padding='same')(x)
  x = BatchNormalization()(x)
  x=MaxPooling2D((2,2))(x)
  x=Dropout(0.25)(x)

  output=Conv2D(31,(3,3),activation='relu',padding='same')(x)

  model=Model(ip,output)
  return model

















def normalize_data(data):
  print("Normalizing Data ...........................")
  print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
  temp = img_as_ubyte(data)
  print(temp.shape, temp.dtype, "min:",temp.min(), "max:",temp.max())
  data = temp/256.0
  print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
  return data	

def resize_npFile(data, image_size, file_save_path=None, save=False):	
	print("Resizing the input data .................",data.shape)
	resize_np = np.zeros((data.shape[0],image_size, image_size, data.shape[3]))
	
	for j in range(data.shape[0]):
		for i in range(data.shape[3]):			
			# cv2.imwrite('org.png',img_as_ubyte(data[j][:,:,i]))
			temp = cv2.resize(data[j][:,:,i],(image_size, image_size), interpolation=cv2.INTER_AREA)	
			resize_np[j][:,:,i] = temp
			# print(data[j].shape, data[j][:,:,i].shape, temp.shape, resize_np[j].shape)			
	if save == True:
		np.save(file_save_path, resize_np)
		print('Successfully Saved .....', file_save_path, resize_np.shape)	
	return resize_np


def main(s):
  image_size = 256
  if s=='train':
    print('numpy file for input data patches already exists. Loading File..................')
    X_data = np.load('/content/drive/My Drive/Colab Notebooks/clean_rgb_010.npy')
    print('Data loaded ........................','/content/drive/My Drive/Colab Notebooks/clean_rgb_010.npy', X_data.shape)	
    print(X_data.shape, X_data.dtype, X_data.min(), X_data.max())
    Y_data = np.load('/content/drive/My Drive/Colab Notebooks/hs_complete_010 .npy')
    print('Data loaded ........................','/content/drive/My Drive/Colab Notebooks/hs_complete_010 .npy', Y_data.shape)		
    print(Y_data.shape, Y_data.dtype, Y_data.min(), Y_data.max())	
    
    X_data = normalize_data(X_data)
    print(X_data.shape, X_data.dtype, X_data.min(), X_data.max())
    X_data = resize_npFile(X_data, image_size, file_save_path=None, save=False)
    Y_data = resize_npFile(Y_data, image_size, file_save_path=None, save=False)
    from sklearn.model_selection import train_test_split
    X_test=X_data[2:9]
    Y_test=Y_data[2:9]
    X_train,  X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2)		
    print("Training data and Ground truth shape.................",X_train.shape,Y_train.shape)
    print('Validation Split Completed...........................', X_val.shape, Y_val.shape)
    print(X_test.shape,Y_test.shape)
    model_to_train = build_net(image_size)
    model_to_train.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0001), metrics=['mae','mse'])
    history = model_to_train.fit(x=X_train, y=Y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2, 
                                 validation_data=(X_val,Y_val),shuffle=True)
    exp=model_to_train.evaluate(X_test,Y_test,verbose=1)
    
    print('test loss, test acc:', exp)
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['mae'], 'r', label='mean_absolute_error loss')
    plt.plot(history.history['mse'], 'g', label='mean_squared_error loss')
    plt.title('Training loss -v')
    plt.legend()
		



if __name__ == '__main__':
	main('train')
