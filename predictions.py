from tensorflow import keras
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
IMG_WIDTH = 30
IMG_HEIGHT = 30

model = keras.models.load_model('model.h5')
y_test = pd.read_csv('test.csv')
y_test.set_index('Path', inplace=True)

data = []
labels = []

for entry in os.listdir('Test'):
    img_path = os.path.join('Test', entry)
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    data.append(img)
    labels.append(y_test['ClassId'][img_path])

X_test = np.array(data)
pred = np.argmax(model.predict(X_test), axis=-1)

# Accuracy with the test data
print(accuracy_score(labels, pred))
