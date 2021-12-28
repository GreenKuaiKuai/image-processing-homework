from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, BatchNormalization


# 訓練樣本目錄和測試樣本目錄
train_dir = './data/train/'
test_dir = './data/validation/'

# 對訓練圖像進行數據增強
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=False,
                                   fill_mode="nearest")

# 對測試圖像進行數據增強
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
model.add(Dense(34))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


#train
his = model.fit_generator(train_flow,
                    steps_per_epoch=100,
                    epochs=50,
                    verbose=1,
                    validation_data=test_flow,
                    validation_steps=500,
                    callbacks=[tbCallBack])


model.save('./weights/char_binary_model.h5')

plt.plot(his.history['acc'])
plt.plot(his.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()