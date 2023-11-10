import numpy as np
import tensorflow as tf
import keras
import cv2
import random
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D, MaxPooling2D,GlobalMaxPooling2D,BatchNormalization
from sklearn.model_selection import KFold
from keras.callbacks import Callback
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
from focal_loss import SparseCategoricalFocalLoss
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

best_model = load_model('best_model.h5')

images, proba, types = load_dataset()
# change label, label will be 0 1 2 3
label_mapping = {0.0: 0, 0.3333333333333333: 1, 0.6666666666666666: 2, 1.0: 3}
for x in range(len(proba)):
    if proba[x] in label_mapping:
        proba[x] = label_mapping[proba[x]]
# get the infomations of images
num_images, height, width = images.shape
# Create a new numpy array to store the resized and denoised images
resized_images = np.empty((num_images, 64, 64), dtype=np.uint8)
# use for loop, resize every image and denoise it
for i in range(num_images):
    resized_images[i] = cv2.resize(images[i], (64,64))
    resized_images[i] = cv2.GaussianBlur(resized_images[i], (3, 3),0)
# split the dataset into train set and test set 75% for train , 25% for test
X_train, X_test, y_train, y_test =  train_test_split(resized_images, proba,train_size=0.75,stratify=proba, random_state=30,shuffle=True)

# Image Enhancement ------- purpose is make the perfomance better, for example, rotation/scaling can reduce the overfiiting and increase the data diversity
# Contrast Enhancement 
new_X_train = X_train.copy()
new_y_train = y_train.copy()
for x in range(len(new_X_train)):
   new_X_train[x] = cv2.equalizeHist(X_train[x])
new_X_train = np.concatenate([X_train, new_X_train])
new_y_train = np.concatenate([y_train, new_y_train])

# flip(horiton + vertical)
temp_X = X_train.copy()
temp_y = y_train.copy()
for i in range(len(temp_X)):
    temp_X[i] = cv2.flip(X_train[i],-1)
new_X_train = np.concatenate([new_X_train, temp_X])
new_y_train = np.concatenate([new_y_train, temp_y])

#Some random rotation 
temp_X = X_train.copy()
temp_y = y_train.copy()
for i in range(len(temp_X)):
    rows, cols = X_train[i].shape
    random_angle = random.randint(-30, 30)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, 1)
    temp_X[i] = cv2.warpAffine(X_train[i], rotation_matrix, (cols, rows))
new_X_train = np.concatenate([new_X_train, temp_X])
new_y_train = np.concatenate([new_y_train, temp_y])
# Now,we got 7968 images for taining; 4524 for class 0; 884 for class 1 ; 320 for class 2 ; 2144 for class 3
# copy 300 images from class 2
class_indices = np.where(new_y_train == 2)[0]
samples_to_copy = np.random.choice(class_indices, size=240, replace=False)
new_X_train = np.concatenate([new_X_train, new_X_train[samples_to_copy]])
new_y_train = np.concatenate([new_y_train, new_y_train[samples_to_copy]])
class_indices = np.where(new_y_train == 1)[0]
samples_to_copy = np.random.choice(class_indices, size=300, replace=False)
new_X_train = np.concatenate([new_X_train, new_X_train[samples_to_copy]])
new_y_train = np.concatenate([new_y_train, new_y_train[samples_to_copy]])
# Fianlly, add some noise on the image, now we get around 10K train set
temp_X = X_train.copy()
temp_y = y_train.copy()
for i in range(len(temp_X)):
    noise = np.random.normal(loc=0, scale=10, size=X_train[i].shape)
    temp_X[i] = np.clip(X_train[i] + noise, 0, 255).astype(np.float32)
new_X_train = np.concatenate([new_X_train, temp_X])
new_y_train = np.concatenate([new_y_train, temp_y])
# make sure all images are float 32
new_X_train = new_X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
# add channel, make sure the image is 4D
new_X_train = new_X_train.reshape(new_X_train.shape[0], new_X_train.shape[1], new_X_train.shape[2],1)
# make sure all pixel is  [0,1]
new_X_train = new_X_train / 255
X_test = X_test / 255

#reshape the X_train set, make sure it's 4D
#Method 1 , Use CNN model 
model = Sequential()
model.add(Conv2D(32, (3, 3),padding='same',activation='relu',input_shape=(64, 64, 1)))
model.add(Conv2D(32, (3, 3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation="softmax"))
# After the model built up, we can compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
             loss=SparseCategoricalFocalLoss(gamma=2),
             metrics=['accuracy'])
#fit
#model.summary()
checkpoint = ModelCheckpoint('try_best_model.h5', 
                             monitor='val_accuracy',  
                             save_best_only=True,  
                             mode='max',  #
                             verbose=1)  
history = model.fit(new_X_train,new_y_train,batch_size=32, epochs=30,validation_split=0.2,callbacks=[checkpoint])
#best_model = load_model('best_model.h5')
# predict
y_pred = model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)
y_test = y_test.astype(int)
# evaluate, f1score / recall /predictions and confusion matrix
f1_score = classification_report(y_test, predicted_labels)
confusion  = confusion_matrix(y_test,predicted_labels)
# print the result
print(f1_score)
print(confusion)
# show the 
#plt.plot(history.history['accuracy'])
#plt.plot(history.history["val_accuracy"])
#plt.title("accuracy")
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.legend(['Train','Test'],loc="upper left")
#plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title("Loss")
#plt.legend(['Train','Test'],loc="upper left")
#plt.show()
