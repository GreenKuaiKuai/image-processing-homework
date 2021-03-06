# -*- coding: utf-8 -*-
"""HW4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t76JMX8NKUXanXWSIGXmuLPyDBOiC3a0

## 訓練 Training
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, BatchNormalization
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 訓練樣本目錄和測試樣本目錄
train_dir = '/content/drive/MyDrive/HW4/train'
test_dir = '/content/drive/MyDrive/HW4/validation'

# 對訓練圖像做正規化
train_pic_gen = ImageDataGenerator(rescale=1./255)

# 對測試圖像做正規化
test_pic_gen = ImageDataGenerator(rescale=1./255)

# 利用 .flow_from_directory 函數生成訓練數據
train_flow = train_pic_gen.flow_from_directory(train_dir,
                        target_size=(224,224),
                        batch_size=64,
                        class_mode='categorical')

# 利用 .flow_from_directory 函數生成測試數據
test_flow = test_pic_gen.flow_from_directory(test_dir,
                      target_size=(224,224),
                      batch_size=64,
                      class_mode='categorical')
print(train_flow.class_indices)

#搭建網路
resize = 224
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/1',
                      histogram_freq= 0,
                      write_graph=True,
                      write_images=True)

model = Sequential()

# level1
model.add(Conv2D(filters=96,kernel_size=(11,11),
        strides=(4,4),padding='valid',
        input_shape=(resize,resize,3),
        activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
            strides=(2,2),
            padding='valid'))

# level_2
model.add(Conv2D(filters=256,kernel_size=(5,5),
        strides=(1,1),padding='same',
        activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
          strides=(2,2),
          padding='valid'))

# layer_3
model.add(Conv2D(filters=384,kernel_size=(3,3),
        strides=(1,1),padding='same',
        activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=384,kernel_size=(3,3),
        strides=(1,1),padding='same',
        activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=356,kernel_size=(3,3),
        activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
          strides=(2,2),padding='valid'))

# layer_4
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
      optimizer='sgd',
      metrics=['accuracy'])

#train
his = model.fit(train_flow,
        epochs=15,
        verbose=1,
        validation_data=test_flow,
        callbacks=[tbCallBack])

model.save('/content/drive/MyDrive/HW4/my_model.h5')

"""## 預測 Testing"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

## 載入模型
model = tf.keras.models.load_model('/content/drive/MyDrive/HW4/my_model.h5')

def predict(img):
  test_img = cv2.resize(img,(224,224))
  test_img = np.asarray(test_img.astype('float32'))
  test_img = test_img/255.
  test_img = test_img.reshape((1,224,224,3))

  prediction = model.predict(test_img)
  index = np.argmax(prediction)
  return dictionary[index]
	
dictionary = {0:'indoor', 1:'outdoor'}
for filename in os.listdir('/content/drive/MyDrive/HW4/test'):
  img = cv2.imread('/content/drive/MyDrive/HW4/test/'+filename)
  ans = predict(img)
  print('圖片:{0}, 預測:{1}'.format(filename, ans))