import numpy as np
import keras
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.utils import to_categorical

# Respectively: image, reflection probability, type. Our classifier needs to classify based on reflection probability
images, proba, types = load_dataset()
# split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(images, proba, test_size=0.25, stratify=proba, random_state=42)
# makesure the type is same [pre-processing data]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# Normalize the train set and test set, make sure the pixel value is between 0 and 1.[pre-processing data]
X_train = X_train / 255
X_test = X_test / 255
# Image Enhancement ------- purpose is make the perfomance better, for example, rotation/scaling can reduce the overfiiting and increase the data diversity
# Use ImageDataGenerator to enhance the image ,there are lots of argument can be changed to influence the performance
# Each argument's explanation is copied from the tensorflow.org (official website) , simply rotate/shift the image
Image_data_gen = ImageDataGenerator(
    rotation_range= 30,     #   random rotation
    width_shift_range=0.1,  #   shift images horizontally (random shift)
    height_shift_range=0.1, #   shift images vertically (random shift)
)
#reshape the X_train set, make sure it's 4D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
#change the y_test to one-hot
y_test = to_categorical(y_test,4)
y_train = to_categorical(y_train,4)
# Fit the image_data_gen
Image_data_gen.fit(X_train)
#Method 1 , Use CNN model
model = Sequential()
# First Layer , start with filters = 32 and the kernel size will be 5x5 due to we got a 300x300 image. Then we choose relu for the activation funtion
# make sure the input_shape is correct, which is 300x300 with 1 channel (gray-level image).
# it's totally same as model.add(Activation(relu'))
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(300,300,1)))  
# Then we add a new conv layer, this time the filters = 64 and the kernel size still be 3x3 
# Then we add a MaxingPolling layer
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3, 3), activation='relu'))
# Then we add a MaxingPolling layer
model.add(MaxPooling2D(2,2))
# Then we add a new cov layer
model.add(Conv2D(128, (3, 3), activation='relu'))
# Then we add a MaxingPolling layer
model.add(MaxPooling2D(2,2))
# Then we use the dropout to reduce the problem of overfitting
model.add(Dropout(0.25))
# After that, we can add a Flatten Layer
model.add(Flatten())
# Moreover, still need a fully connected layer
model.add(Dense(64,activation='relu'))
# Last, the output layer
model.add(Dropout(0.25))
model.add(Dense(4,activation='softmax'))

# After the model built up, we can compile the model
model.compile(optimizer='adam', 
             loss='categorical_crossentropy',
             metrics=['accuracy'])


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
#model.summary()