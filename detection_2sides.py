import cv2
import numpy as np
datasetdir = './Train'
import os
os.chdir(datasetdir)
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import subprocess, sys

resnet50 = keras.applications.resnet50
conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in conv_model.layers:
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(3, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

full_model.load_weights('resnet50.h5')
labels = ["Maciek", "Hubert", "Krzysiek"]

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Maciek', 'Hubert', 'Krzysiek']
cam = cv2.VideoCapture(0)
cam.set(3, 1920)
cam.set(4, 1080)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
counter=0
counter1=0
while True:
    ret, img = cam.read()

    # Pobranie szerokości i wysokości obrazu
    height, width, _ = img.shape

    # Definicja regionów do detekcji ludzi (lewa i prawa połowa obrazu)
    left_region = (0, 0, int(width / 2), height)
    right_region = (int(width / 2), 0, int(width / 2), height)
    # Wycięcie lewej i prawej połowy obrazu
    left_image = img[:, :int(width / 2)]
    right_image = img[:, int(width / 2):]
    # Detekcja ludzi w lewej połowie obrazu
    gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    gray2 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    faces2 = faceCascade.detectMultiScale(
        gray2,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("./x.jpg", gray[y:y + h, x:x + w])
        img_path = './x.jpg'
        z = image.load_img(img_path, target_size=(224, 224))
        i = image.img_to_array(z)
        i = np.expand_dims(i, axis=0)
        i = preprocess_input(i)
        id = np.argmax(full_model.predict(i))
        confidence = "  {0}%".format(round(100))

        print(labels[id])
        es=labels[id]+' '+str(np.max(full_model.predict(i))*100)+'%'
        cv2.putText((img),es,(x + 5, y - 5),
            font,1,(255, 255, 255),1,2
        )
    for (x, y, w, h) in faces2:
        cv2.rectangle(img, (x+int(width / 2), y), (x + w+int(width / 2), y + h), (0, 255, 0), 2)
        cv2.imwrite("./x.jpg", gray2[y:y + h, x:x + w])
        img_path = './x.jpg'
        z = image.load_img(img_path, target_size=(224, 224))
        i = image.img_to_array(z)
        i = np.expand_dims(i, axis=0)
        i = preprocess_input(i)
        id = np.argmax(full_model.predict(i))
        confidence = "  {0}%".format(round(100))

        print(labels[id])
        es=labels[id]+' '+str(np.max(full_model.predict(i))*100)+'%'
        cv2.putText((img),es,(x + 5+int(width / 2), y - 5),
            font,1,(255, 255, 255),1,2
        )
    if len(faces):
        counter=0
    else:
        counter+=1
    if len(faces2):
        counter1=0
    else:
        counter1+=1
    if counter>30:

        p = subprocess.Popen('.\\ncat.exe 192.168.41.45 4444',stdout=sys.stdout)
        p.communicate()
    if counter1>30:
        p = subprocess.Popen('.\\ncat.exe 192.168.41.103 4444', stdout=sys.stdout)
        p.communicate()
    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n Exit")
cam.release()
cv2.destroyAllWindows()