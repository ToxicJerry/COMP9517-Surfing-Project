import numpy
import keras
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.utils import to_categorical
#分别为：图像，反射概率，类型。  我们的分类器需要根据反射概率进行分类
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
#fit
model.fit(Image_data_gen.flow(X_train,y_train, batch_size=32),epochs=5,steps_per_epoch=len(X_train)//32,validation_data=(X_test,y_test))
#model.summary()