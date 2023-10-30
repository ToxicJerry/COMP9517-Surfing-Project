import numpy
import keras
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D, MaxPooling2D
#分别为：图像，反射概率，类型。  我们的分类器需要根据反射概率进行分类
images, proba, types = load_dataset()
# split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(images, proba, test_size=0.25, stratify=proba, random_state=42)
# Normalize the train set and test set, make sure the pixel value is between 0 and 1.[pre-processing data]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
#Method 1 , Use CNN model
model = Sequential()
# First Layer , start with filters = 32 and the kernel size will be 3x3. Then we choose relu for the activation funtion
# it's totally same as model.add(Activation(relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
# Then we add a new conv layer, this time the filters = 64 and the kernel size still be 3x3 
model.add(Conv2D(64, (3, 3), activation='relu'))
# Then we add a MaxingPolling layer
model.add(MaxPooling2D(2,2))
# Then we use the dropout to reduce the problem of overfitting
model.add(Dropout(0.25))
# After that, we can add a Flatten Layer
model.add(Flatten())
# Moreover, still need a fully connected layer
model.add(Dense(64,activation='relu'))
# Last, the output layer
model.add(Dense(4,activation='softmax'))

# After the model built up, we can compile the model
model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
#fit
model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))