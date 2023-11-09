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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from sklearn.preprocessing import label_binarize

# Respectively: image, reflection probability, type. Our classifier needs to classify based on reflection probability
images, proba, types = load_dataset()
labels = np.unique(proba)
type1 = 0.0
type2 = 0.3333333333333333
type3 = 0.6666666666666666
type4 = 1.0
for x in range(len(proba)):
    if proba[x] == type1:
        proba[x] = 0
        proba[x] = proba[x].astype(int)
        continue
    if proba[x] == type2:
        proba[x] = 1
        proba[x] = proba[x].astype(int)
        continue
    if proba[x] == type3:
        proba[x] = 2
        proba[x] = proba[x].astype(int)
    if proba[x] == type4:
        proba[x] = 3
        proba[x] = proba[x].astype(int)
# get the infomations of images
num_images, height, width = images.shape
# Create a new numpy array to store the resized and denoised images
resized_images = np.empty((num_images, 128, 128), dtype=np.float32)
# use for loop, resize every image and denoise it
for i in range(num_images):
    resized_images[i] = cv2.resize(images[i], (128, 128))
    resized_images[i] = cv2.blur(resized_images[i], (5, 5))
# split the dataset into train set and test set 75% for train , 12.5% for validation , 12.5 for test
X_train, X_test, y_train, y_test =  train_test_split(resized_images, proba,train_size=0.75,stratify=proba, random_state=7,shuffle=True)
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
# Fit the image_data_gen
Image_data_gen.fit(X_train)
#Method 1 , Use CNN model , and the main idea is coming from Alexnet. Alexnet got 8 layers totally (5 conv layer + 3 fully connect layer)
model = Sequential()
# First Layer , Alexnet uses filters = 96 and the kernel size is 11x11 , stride = 4. 
# However, I do not think the paramater should be exactly same Alexnet. we use 64 + 7x7 with 2 stride.
# make sure the input_shape is correct, which is 128x128 with 1 channel (gray-level image).
# activation= "relu",it's totally same as model.add(Activation(relu'))
model.add(Conv2D(64, (3, 3),activation='relu',input_shape=(128,128,1)))  
# For the 2nd layer, Alexnet uses maxpool with size 3x3, but I think 2x2 is better for our task
#model.add(Conv2D(32, (3, 3),activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2)))
# Next, it's the 2nd conv layer, Alexnet used 256 and 5x5 as paramater
# However, I will change it to 128 and 5x5
#model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
# Next, add a maxpooling
#model.add(MaxPooling2D(pool_size=(2,2)))
# 3rd conv layer , no maxpooling for this convlayer
#model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
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
#model.add(Dense(256,activation="relu"))
# a large dropout, reduce the overfitting
# second fully connet
model.add(Dense(256,activation='relu'))
# Third fullyconnected layer, for classfier
#model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dense(4,activation="softmax"))
# After the model built up, we can compile the model
#opt = SGD(learning_rate=0.001)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
#fit
history = model.fit(Image_data_gen.flow(X_train,y_train,batch_size=8),epochs=5,steps_per_epoch=len(X_train)//8,validation_data=(X_test,y_test))
#model.summary()
y_pred = model.predict(X_test)
predicted_labels = np.argmax(y_pred, axis=1)
#predicted_class_indices = np.argmax(y_pred, axis=1)
#one_hot_predictions = to_categorical(predicted_class_indices, num_classes=4)
y_test = y_test.astype(int)
f1_score = classification_report(y_test, predicted_labels)
confusion  = confusion_matrix(y_test,predicted_labels)
print("Confusion Matrix for All Images:")
print(f1_score)
print(confusion)
labels = ["Class 0", "Class 1", "Class 2", "Class 3"]  
for i in range(len(labels)):
    for j in range(len(labels)):
        label_str = f"{labels[i]} (Actual) vs {labels[j]} (Predicted)"
        print(f"{label_str}: {confusion[i, j]}")
# Confusion Matrix for All Images:
#                 precision    recall  f1-score   support

#            0       0.67      0.89      0.77       377
#            1       0.00      0.00      0.00        74
#            2       0.00      0.00      0.00        26
#            3       0.65      0.56      0.60       179

#     accuracy                           0.67       656
#    macro avg       0.33      0.36      0.34       656
# weighted avg       0.56      0.67      0.61       656

# [[337   0   0  40]
#  [ 65   0   0   9]
#  [ 21   0   0   5]
#  [ 78   0   0 101]]
# Class 0 (Actual) vs Class 0 (Predicted): 337
# Class 0 (Actual) vs Class 1 (Predicted): 0
# Class 0 (Actual) vs Class 2 (Predicted): 0
# Class 0 (Actual) vs Class 3 (Predicted): 40
# Class 1 (Actual) vs Class 0 (Predicted): 65
# Class 1 (Actual) vs Class 1 (Predicted): 0
# Class 1 (Actual) vs Class 2 (Predicted): 0
# Class 1 (Actual) vs Class 3 (Predicted): 9
# Class 2 (Actual) vs Class 0 (Predicted): 21
# Class 2 (Actual) vs Class 1 (Predicted): 0
# Class 2 (Actual) vs Class 2 (Predicted): 0
# Class 2 (Actual) vs Class 3 (Predicted): 5
# Class 3 (Actual) vs Class 0 (Predicted): 78
# Class 3 (Actual) vs Class 1 (Predicted): 0
# Class 3 (Actual) vs Class 2 (Predicted): 0
# Class 3 (Actual) vs Class 3 (Predicted): 101


# Select only the monocrystalline image and calculate the confusion matrix
monocrystalline_indices = np.where(y_test == 0)
X_test_monocrystalline = X_test[monocrystalline_indices]
y_test_monocrystalline = y_test[monocrystalline_indices]
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
# Confusion Matrix for Monocrystalline Images:
#                precision    recall  f1-score   support

#            0       1.00      0.89      0.94       377
#            3       0.00      0.00      0.00         0

#     accuracy                           0.89       377
#    macro avg       0.50      0.45      0.47       377
# weighted avg       1.00      0.89      0.94       377

# [[337   0   0  40]
#  [  0   0   0   0]
#  [  0   0   0   0]
#  [  0   0   0   0]]
# Class 0 (Actual) vs Class 0 (Predicted): 337
# Class 0 (Actual) vs Class 1 (Predicted): 0
# Class 1 (Actual) vs Class 0 (Predicted): 0
# Class 1 (Actual) vs Class 1 (Predicted): 0

# Select only the polycrystalline images and calculate the confusion matrix
polycrystalline_indices = np.where(y_test == 1)
X_test_polycrystalline = X_test[polycrystalline_indices]
y_test_polycrystalline = y_test[polycrystalline_indices]
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
# Confusion Matrix for Polycrystalline Images:
#                precision    recall  f1-score   support

#            0       0.00      0.00      0.00       0.0
#            1       0.00      0.00      0.00      74.0
#            3       0.00      0.00      0.00       0.0

#     accuracy                           0.00      74.0
#    macro avg       0.00      0.00      0.00      74.0
# weighted avg       0.00      0.00      0.00      74.0

# [[ 0  0  0  0]
#  [65  0  0  9]
#  [ 0  0  0  0]
#  [ 0  0  0  0]]
# Class 0 (Actual) vs Class 0 (Predicted): 0
# Class 0 (Actual) vs Class 1 (Predicted): 0
# Class 0 (Actual) vs Class 2 (Predicted): 0
# Class 0 (Actual) vs Class 3 (Predicted): 0
# Class 1 (Actual) vs Class 0 (Predicted): 65
# Class 1 (Actual) vs Class 1 (Predicted): 0
# Class 1 (Actual) vs Class 2 (Predicted): 0
# Class 1 (Actual) vs Class 3 (Predicted): 9
# Class 2 (Actual) vs Class 0 (Predicted): 0
# Class 2 (Actual) vs Class 1 (Predicted): 0
# Class 2 (Actual) vs Class 2 (Predicted): 0
# Class 2 (Actual) vs Class 3 (Predicted): 0
# Class 3 (Actual) vs Class 0 (Predicted): 0
# Class 3 (Actual) vs Class 1 (Predicted): 0
# Class 3 (Actual) vs Class 2 (Predicted): 0
# Class 3 (Actual) vs Class 3 (Predicted): 0


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
