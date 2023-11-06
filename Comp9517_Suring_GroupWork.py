import numpy as np
import keras
import cv2
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.utils import to_categorical

# Respectively: image, reflection probability, type. Our classifier needs to classify based on reflection probability
images, proba, types = load_dataset()
# get the infomations of images
num_images, height, width = images.shape
# Create a new numpy array to store the resized and denoised images
resized_images = np.empty((num_images, 128, 128), dtype=np.uint8)
# use for loop, resize every image and denoise it
for i in range(num_images):
    resized_images[i] = cv2.resize(images[i], (128, 128))
    resized_images[i] = cv2.blur(resized_images[i], (5, 5))
# split the dataset into train set and test set
X_train, X_test, y_train, y_test =  train_test_split(resized_images, proba,test_size=0.25,stratify=proba, random_state=90)
# makesure the type is same [pre-processing data]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# Normalize the train set and test set, make sure the pixel value is between 0 and 1.[pre-processing data]
X_train = X_train / 255
X_test = X_test / 255
# Image Enhancement ------- purpose is make the perfomance better, for example, rotation/scaling can reduce the overfiiting and increase the data diversity
# Use ImageDataGenerator to enhance the image ,there are lots of argument can be changed to influence the performance
# Each argument's explanation is copied from the tensorflow.org (official website) , simply rotate/shift/flip the image
Image_data_gen = ImageDataGenerator(
    rotation_range= 30,     #   random rotation
    width_shift_range=0.2,  #   shift images horizontally (random shift)
    height_shift_range=0.2, #   shift images vertically (random shift)
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True #   flip images horizontally (random flip)
)
#reshape the X_train set, make sure it's 4D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
#change the y_test to one-hot
y_test = to_categorical(y_test,4)
y_train = to_categorical(y_train,4)
# Fit the image_data_gen
Image_data_gen.fit(X_train)
#Method 1 , Use CNN model , and the main idea is coming from Alexnet. Alexnet got 8 layers totally (5 conv layer + 3 fully connect layer)
model = Sequential()
# First Layer , Alexnet uses filters = 96 and the kernel size is 11x11 , stride = 4. 
# However, I do not think the paramater should be exactly same Alexnet. we use 64 + 7x7 with 2 stride.
# make sure the input_shape is correct, which is 128x128 with 1 channel (gray-level image).
# activation= "relu",it's totally same as model.add(Activation(relu'))
model.add(Conv2D(32, (3, 3),activation='relu',input_shape=(128,128,1)))  
# For the 2nd layer, Alexnet uses maxpool with size 3x3, but I think 2x2 is better for our task
model.add(Conv2D(32, (3, 3),activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2)))
# Next, it's the 2nd conv layer, Alexnet used 256 and 5x5 as paramater
# However, I will change it to 128 and 5x5
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
# Next, add a maxpooling
model.add(MaxPooling2D(pool_size=(2,2)))
# 3rd conv layer , no maxpooling for this convlayer
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# 4th conv layer , no maxpooling for this convlayer
#model.add(Conv2D(384,(5,5),activation='relu',padding='same'))
# the last conv layer , maxpooling needed
#model.add(Conv2D(256,(5,5),activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#model.add(Conv2D(256,(5,5),activation='relu',padding='same'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=2))
# After the conv layer , Alexnet still got 3 fullyconnect layer
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
# a large dropout, reduce the overfitting
# second fully connet
#model.add(Dense(4096,activation='relu'))
# Third fullyconnected layer, for classfier
#model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4,activation="softmax"))
# After the model built up, we can compile the model
opt = SGD(learning_rate=0.01)
model.compile(optimizer=opt, 
             loss='categorical_crossentropy',
             metrics=['accuracy'])
<<<<<<< HEAD
#fit
history = model.fit(Image_data_gen.flow(X_train,y_train,batch_size=8),epochs=30,steps_per_epoch=len(X_train)//8,validation_data=(X_test,y_test))
#model.summary()
plt.plot(history.history['accuracy'])
plt.plot(history.history["val_accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['Train','Test'],loc="upper left")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(['Train','Test'],loc="upper left")
plt.show()
=======


# The following code begins model training：

# 1. Test all data sets
# model.fit(Image_data_gen.flow(X_train,y_train, batch_size=32),epochs=5,steps_per_epoch=len(X_train)//32,validation_data=(X_test,y_test))
# Result:
# (envName) masisi@MacBook-Pro COMP9517-Surfing-Project % python Comp9517_Suring_GroupWork.py
# Epoch 1/5
# 61/61 [==============================] - 123s 2s/step - loss: 0.6914 - accuracy: 0.7169 - val_loss: 0.5057 - val_accuracy: 0.7500
# Epoch 2/5
# 61/61 [==============================] - 122s 2s/step - loss: 0.5278 - accuracy: 0.7634 - val_loss: 0.5440 - val_accuracy: 0.7561
# Epoch 3/5
# 61/61 [==============================] - 122s 2s/step - loss: 0.5288 - accuracy: 0.7655 - val_loss: 0.5040 - val_accuracy: 0.7637
# Epoch 4/5
# 61/61 [==============================] - 137s 2s/step - loss: 0.5105 - accuracy: 0.7831 - val_loss: 0.4592 - val_accuracy: 0.8110
# Epoch 5/5
# 61/61 [==============================] - 138s 2s/step - loss: 0.5039 - accuracy: 0.7810 - val_loss: 0.4790 - val_accuracy: 0.7698

# 2. Test 2/3 data set
# # Split the data set into 2/3 training data and 1/3 test data
# X_train_2_3, X_test_1_3, y_train_2_3, y_test_1_3 = train_test_split(images, proba, test_size=1/3, stratify=proba, random_state=42)
# # Change the dimension to 4D
# X_train_2_3 = X_train_2_3.reshape(X_train_2_3.shape[0], X_train_2_3.shape[1], X_train_2_3.shape[2], 1)
# X_test_1_3 = X_test_1_3.reshape(X_test_1_3.shape[0], X_test_1_3.shape[1], X_test_1_3.shape[2], 1)
# # Ensure labels are one-hot encoded
# y_train_2_3 = to_categorical(y_train_2_3, num_classes=4)
# y_test_1_3 = to_categorical(y_test_1_3, num_classes=4)
# # Operational model training
# model.fit(Image_data_gen.flow(X_train_2_3, y_train_2_3, batch_size=32), epochs=5, steps_per_epoch=len(X_train_2_3)//32, validation_data=(X_test_1_3, y_test_1_3))
# Result:
# (envName) masisi@MacBook-Pro COMP9517-Surfing-Project % python Comp9517_Suring_GroupWork.py
# Epoch 1/5
# 54/54 [==============================] - 119s 2s/step - loss: 30.9215 - accuracy: 0.6278 - val_loss: 0.7718 - val_accuracy: 0.6629
# Epoch 2/5
# 54/54 [==============================] - 114s 2s/step - loss: 0.6766 - accuracy: 0.6942 - val_loss: 0.5994 - val_accuracy: 0.7303
# Epoch 3/5
# 54/54 [==============================] - 113s 2s/step - loss: 0.6263 - accuracy: 0.7135 - val_loss: 0.6263 - val_accuracy: 0.7303
# Epoch 4/5
# 54/54 [==============================] - 114s 2s/step - loss: 0.6359 - accuracy: 0.7082 - val_loss: 0.5779 - val_accuracy: 0.7383
# Epoch 5/5
# 54/54 [==============================] - 114s 2s/step - loss: 0.5913 - accuracy: 0.7187 - val_loss: 0.6211 - val_accuracy: 0.6720

# 3. Test 1/3 data set
X_train_1_3, X_test_2_3, y_train_1_3, y_test_2_3 = train_test_split(images, proba, test_size=2/3, stratify=proba, random_state=42)
# NumpyArrayIterator requires input data to have four dimensions (number of samples, image height, image width, and number of channels), and X_train_1_3 is three-dimensional. 
# Therefore, it is necessary to change X_train_1_3 from three-dimensional to four-dimensional by adding the number of channels in the last dimension.
X_train_1_3 = np.expand_dims(X_train_1_3, axis=-1)
X_test_2_3 = np.expand_dims(X_test_2_3, axis=-1)
# Labels y_train_1_3 and y_test_2_3 have been thermally coded
y_train_1_3 = to_categorical(y_train_1_3, num_classes=4)
y_test_2_3 = to_categorical(y_test_2_3, num_classes=4)
model.fit(Image_data_gen.flow(X_train_1_3, y_train_1_3, batch_size=32), epochs=5, steps_per_epoch=len(X_train_1_3)//32, validation_data=(X_test_2_3, y_test_2_3))
# Result:
# (envName) masisi@MacBook-Pro COMP9517-Surfing-Project % python Comp9517_Suring_GroupWork.py
# Epoch 1/5
# 27/27 [==============================] - 77s 3s/step - loss: 9.5415 - accuracy: 0.5950 - val_loss: 0.6131 - val_accuracy: 0.7274
# Epoch 2/5
# 27/27 [==============================] - 76s 3s/step - loss: 0.6194 - accuracy: 0.7150 - val_loss: 0.6274 - val_accuracy: 0.7274
# Epoch 3/5
# 27/27 [==============================] - 76s 3s/step - loss: 0.6326 - accuracy: 0.6936 - val_loss: 0.6216 - val_accuracy: 0.7274
# Epoch 4/5
# 27/27 [==============================] - 76s 3s/step - loss: 0.6248 - accuracy: 0.7055 - val_loss: 0.6246 - val_accuracy: 0.7274
# Epoch 5/5
# 27/27 [==============================] - 76s 3s/step - loss: 0.6013 - accuracy: 0.7067 - val_loss: 0.5832 - val_accuracy: 0.7274
>>>>>>> 31b5fb32b25ff0f01052d703fa88c719eae8a527
#model.summary()