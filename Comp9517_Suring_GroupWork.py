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
# Fianlly, add some noise on the original image, now we get around 10K train set
temp_X = X_train.copy()
temp_y = y_train.copy()
for i in range(len(temp_X)):
    noise = np.random.normal(loc=0, scale=15, size=X_train[i].shape)
    temp_X[i] = np.clip(X_train[i] + noise, 0, 255).astype(np.float32)
new_X_train = np.concatenate([new_X_train, temp_X])
new_y_train = np.concatenate([new_y_train, temp_y])
# add some extra class 1 and 2 images
class_indices = np.where(new_y_train == 2)[0]
samples_to_copy = np.random.choice(class_indices, size=240, replace=False)
new_X_train = np.concatenate([new_X_train, new_X_train[samples_to_copy]])
new_y_train = np.concatenate([new_y_train, new_y_train[samples_to_copy]])
class_indices = np.where(new_y_train == 1)[0]
samples_to_copy = np.random.choice(class_indices, size=200, replace=False)
new_X_train = np.concatenate([new_X_train, new_X_train[samples_to_copy]])
new_y_train = np.concatenate([new_y_train, new_y_train[samples_to_copy]])
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
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(512,activation='relu'))
model.add(Dense(4,activation="softmax"))
# After the model built up, we can compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
             loss=SparseCategoricalFocalLoss(gamma=2),
             metrics=['accuracy'])
#fit
#model.summary()
checkpoint = ModelCheckpoint('big_best_model.h5', 
                             monitor='val_accuracy',  
                             save_best_only=True,  
                             mode='max',  #
                             verbose=1)  
#history = model.fit(new_X_train,new_y_train,batch_size=64, epochs=30,validation_split=0.2,callbacks=[checkpoint])
best_model = load_model('big_best_model.h5')
# predict
y_pred = best_model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)
y_test = y_test.astype(int)
# evaluate, f1score / recall /predictions and confusion matrix
f1_score = classification_report(y_test, predicted_labels)
confusion  = confusion_matrix(y_test,predicted_labels)

#plt.plot(history.history['accuracy'])
#plt.plot(history.history["val_accuracy"])
#plt.title("accuracy")
#plt.xlabel("epoch")
#plt.ylabel("accuracy")
#plt.legend(['Train','Test'],loc="upper left")
#plt.show()
print("Confusion Matrix for All Images:")
print(f1_score)
print(confusion)
labels = ["Class 0", "Class 1", "Class 2", "Class 3"]  
for i in range(len(labels)):
    for j in range(len(labels)):
        label_str = f"{labels[i]} (Actual) vs {labels[j]} (Predicted)"
        print(f"{label_str}: {confusion[i, j]}")
# Test Result:
# Confusion Matrix for All Images:
#               precision    recall  f1-score   support

#            0       0.75      0.88      0.81       377
#            1       0.53      0.41      0.46        74
#            2       0.12      0.04      0.06        26
#            3       0.74      0.62      0.67       179

#     accuracy                           0.72       656
#    macro avg       0.53      0.49      0.50       656
# weighted avg       0.70      0.72      0.70       656

# [[331  14   3  29]
#  [ 38  30   1   5]
#  [ 16   3   1   6]
#  [ 55  10   3 111]]
# Class 0 (Actual) vs Class 0 (Predicted): 331
# Class 0 (Actual) vs Class 1 (Predicted): 14
# Class 0 (Actual) vs Class 2 (Predicted): 3
# Class 0 (Actual) vs Class 3 (Predicted): 29
# Class 1 (Actual) vs Class 0 (Predicted): 38
# Class 1 (Actual) vs Class 1 (Predicted): 30
# Class 1 (Actual) vs Class 2 (Predicted): 1
# Class 1 (Actual) vs Class 3 (Predicted): 5
# Class 2 (Actual) vs Class 0 (Predicted): 16
# Class 2 (Actual) vs Class 1 (Predicted): 3
# Class 2 (Actual) vs Class 2 (Predicted): 1
# Class 2 (Actual) vs Class 3 (Predicted): 6
# Class 3 (Actual) vs Class 0 (Predicted): 55
# Class 3 (Actual) vs Class 1 (Predicted): 10
# Class 3 (Actual) vs Class 2 (Predicted): 3
# Class 3 (Actual) vs Class 3 (Predicted): 111


# Select only the monocrystalline image and calculate the confusion matrix
monocrystalline_indices = np.where(types == 'mono')[0]

X_test_monocrystalline = X_test[monocrystalline_indices[monocrystalline_indices < len(X_test)]]
y_test_monocrystalline = y_test[monocrystalline_indices[monocrystalline_indices < len(y_test)]]
y_pred_monocrystalline = best_model.predict(X_test_monocrystalline)
predicted_labels_monocrystalline = np.argmax(y_pred_monocrystalline, axis=1)
# confusion_monocrystalline = confusion_matrix(y_test_monocrystalline, predicted_labels_monocrystalline)
f1_score_monocrystalline = classification_report(y_test_monocrystalline, predicted_labels_monocrystalline)
confusion_monocrystalline = np.zeros((4, 4), dtype=int)
for actual, predicted in zip(y_test_monocrystalline, predicted_labels_monocrystalline):
    confusion_monocrystalline[actual, predicted] += 1
print("Confusion Matrix for Monocrystalline Images:")
print(f1_score_monocrystalline)
print(confusion_monocrystalline)
labels_mono = ["Class 0", "Class 1"]  # the class label with a single crystal
for i in range(len(labels_mono)):
    for j in range(len(labels_mono)):
        label_str = f"{labels_mono[i]} (Actual) vs {labels_mono[j]} (Predicted)"
        print(f"{label_str}: {confusion_monocrystalline[i, j]}")
# Test Result:
# Confusion Matrix for Monocrystalline Images:
#               precision    recall  f1-score   support

#            0       0.74      0.84      0.79       135
#            1       0.46      0.38      0.41        32
#            2       0.00      0.00      0.00         9
#            3       0.72      0.67      0.69        64

#     accuracy                           0.70       240
#    macro avg       0.48      0.47      0.47       240
# weighted avg       0.67      0.70      0.68       240

# [[114   9   0  12]
#  [ 17  12   0   3]
#  [  5   2   0   2]
#  [ 18   3   0  43]]
# Class 0 (Actual) vs Class 0 (Predicted): 114
# Class 0 (Actual) vs Class 1 (Predicted): 9
# Class 1 (Actual) vs Class 0 (Predicted): 17
# Class 1 (Actual) vs Class 1 (Predicted): 12

# Select only the polycrystalline images and calculate the confusion matrix
polycrystalline_indices = np.where(types == 'poly')[0]
X_test_polycrystalline = X_test[polycrystalline_indices[polycrystalline_indices < len(X_test)]]
y_test_polycrystalline = y_test[polycrystalline_indices[polycrystalline_indices < len(y_test)]]
y_pred_polycrystalline = best_model.predict(X_test_polycrystalline)
predicted_labels_polycrystalline = np.argmax(y_pred_polycrystalline, axis=1)
# confusion_polycrystalline = confusion_matrix(y_test_polycrystalline, predicted_labels_polycrystalline)
f1_score_polycrystalline = classification_report(y_test_polycrystalline, predicted_labels_polycrystalline)
confusion_polycrystalline = np.zeros((4, 4), dtype=int)
for actual, predicted in zip(y_test_polycrystalline, predicted_labels_polycrystalline):
    confusion_polycrystalline[actual, predicted] += 1
print("Confusion Matrix for Polycrystalline Images:")
print(f1_score_polycrystalline)
print(confusion_polycrystalline)
labels_poly = ["Class 0", "Class 1", "Class 2", "Class 3"]  # the class label for polycrystals
for i in range(len(labels_poly)):
    for j in range(len(labels_poly)):
        label_str = f"{labels_poly[i]} (Actual) vs {labels_poly[j]} (Predicted)"
        print(f"{label_str}: {confusion_polycrystalline[i, j]}")
# Test Result:
# Confusion Matrix for Polycrystalline Images:
#               precision    recall  f1-score   support

#            0       0.76      0.90      0.82       242
#            1       0.58      0.43      0.49        42
#            2       0.12      0.06      0.08        17
#            3       0.75      0.59      0.66       115

#     accuracy                           0.73       416
#    macro avg       0.55      0.49      0.51       416
# weighted avg       0.71      0.73      0.71       416

# [[217   5   3  17]
#  [ 21  18   1   2]
#  [ 11   1   1   4]
#  [ 37   7   3  68]]
# Class 0 (Actual) vs Class 0 (Predicted): 217
# Class 0 (Actual) vs Class 1 (Predicted): 5
# Class 0 (Actual) vs Class 2 (Predicted): 3
# Class 0 (Actual) vs Class 3 (Predicted): 17
# Class 1 (Actual) vs Class 0 (Predicted): 21
# Class 1 (Actual) vs Class 1 (Predicted): 18
# Class 1 (Actual) vs Class 2 (Predicted): 1
# Class 1 (Actual) vs Class 3 (Predicted): 2
# Class 2 (Actual) vs Class 0 (Predicted): 11
# Class 2 (Actual) vs Class 1 (Predicted): 1
# Class 2 (Actual) vs Class 2 (Predicted): 1
# Class 2 (Actual) vs Class 3 (Predicted): 4
# Class 3 (Actual) vs Class 0 (Predicted): 37
# Class 3 (Actual) vs Class 1 (Predicted): 7
# Class 3 (Actual) vs Class 2 (Predicted): 3
# Class 3 (Actual) vs Class 3 (Predicted): 68

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
