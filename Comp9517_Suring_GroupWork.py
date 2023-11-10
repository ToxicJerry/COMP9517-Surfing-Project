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

#            0       0.69      0.90      0.78       377
#            1       0.35      0.15      0.21        74
#            2       0.10      0.08      0.09        26
#            3       0.78      0.50      0.61       179

#     accuracy                           0.67       656
#    macro avg       0.48      0.41      0.42       656
# weighted avg       0.65      0.67      0.64       656

# [[340   9   9  19]
#  [ 57  11   4   2]
#  [ 18   2   2   4]
#  [ 76   9   5  89]]
# Class 0 (Actual) vs Class 0 (Predicted): 340
# Class 0 (Actual) vs Class 1 (Predicted): 9
# Class 0 (Actual) vs Class 2 (Predicted): 9
# Class 0 (Actual) vs Class 3 (Predicted): 19
# Class 1 (Actual) vs Class 0 (Predicted): 57
# Class 1 (Actual) vs Class 1 (Predicted): 11
# Class 1 (Actual) vs Class 2 (Predicted): 4
# Class 1 (Actual) vs Class 3 (Predicted): 2
# Class 2 (Actual) vs Class 0 (Predicted): 18
# Class 2 (Actual) vs Class 1 (Predicted): 2
# Class 2 (Actual) vs Class 2 (Predicted): 2
# Class 2 (Actual) vs Class 3 (Predicted): 4
# Class 3 (Actual) vs Class 0 (Predicted): 76
# Class 3 (Actual) vs Class 1 (Predicted): 9
# Class 3 (Actual) vs Class 2 (Predicted): 5
# Class 3 (Actual) vs Class 3 (Predicted): 89


# Select only the monocrystalline image and calculate the confusion matrix
monocrystalline_indices = np.where(types == 'mono')[0]
X_test_monocrystalline = X_test[monocrystalline_indices[monocrystalline_indices < len(X_test)]]
y_test_monocrystalline = y_test[monocrystalline_indices[monocrystalline_indices < len(y_test)]]
y_pred_monocrystalline = model.predict(X_test_monocrystalline)
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

#            0       0.67      0.89      0.76       135
#            1       0.44      0.12      0.20        32
#            2       0.10      0.11      0.11         9
#            3       0.73      0.47      0.57        64

#     accuracy                           0.65       240
#    macro avg       0.49      0.40      0.41       240
# weighted avg       0.63      0.65      0.61       240

# [[120   2   4   9]
#  [ 25   4   3   0]
#  [  6   0   1   2]
#  [ 29   3   2  30]]
# Class 0 (Actual) vs Class 0 (Predicted): 120
# Class 0 (Actual) vs Class 1 (Predicted): 2
# Class 1 (Actual) vs Class 0 (Predicted): 25
# Class 1 (Actual) vs Class 1 (Predicted): 4

# Select only the polycrystalline images and calculate the confusion matrix
polycrystalline_indices = np.where(types == 'poly')[0]
X_test_polycrystalline = X_test[polycrystalline_indices[polycrystalline_indices < len(X_test)]]
y_test_polycrystalline = y_test[polycrystalline_indices[polycrystalline_indices < len(y_test)]]
y_pred_polycrystalline = model.predict(X_test_polycrystalline)
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

#            0       0.71      0.91      0.80       242
#            1       0.32      0.17      0.22        42
#            2       0.10      0.06      0.07        17
#            3       0.81      0.51      0.63       115

#     accuracy                           0.69       416
#    macro avg       0.48      0.41      0.43       416
# weighted avg       0.67      0.69      0.66       416

# [[220   7   5  10]
#  [ 32   7   1   2]
#  [ 12   2   1   2]
#  [ 47   6   3  59]]
# Class 0 (Actual) vs Class 0 (Predicted): 220
# Class 0 (Actual) vs Class 1 (Predicted): 7
# Class 0 (Actual) vs Class 2 (Predicted): 5
# Class 0 (Actual) vs Class 3 (Predicted): 10
# Class 1 (Actual) vs Class 0 (Predicted): 32
# Class 1 (Actual) vs Class 1 (Predicted): 7
# Class 1 (Actual) vs Class 2 (Predicted): 1
# Class 1 (Actual) vs Class 3 (Predicted): 2
# Class 2 (Actual) vs Class 0 (Predicted): 12
# Class 2 (Actual) vs Class 1 (Predicted): 2
# Class 2 (Actual) vs Class 2 (Predicted): 1
# Class 2 (Actual) vs Class 3 (Predicted): 2
# Class 3 (Actual) vs Class 0 (Predicted): 47
# Class 3 (Actual) vs Class 1 (Predicted): 6
# Class 3 (Actual) vs Class 2 (Predicted): 3
# Class 3 (Actual) vs Class 3 (Predicted): 59

plt.plot(history.history['accuracy'])
plt.plot(history.history["val_accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['Train','Test'],loc="upper left")
plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title("Loss")
#plt.legend(['Train','Test'],loc="upper left")
#plt.show()
