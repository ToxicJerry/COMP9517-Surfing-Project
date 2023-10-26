import numpy
from utils.elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
#分别为：图像，反射概率，类型。  我们的分类器需要根据反射概率进行分类
images, proba, types = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(images, proba, test_size=0.25, random_state=42)
print(len(X_train))