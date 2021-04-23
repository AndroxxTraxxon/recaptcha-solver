from keras.models import model_from_json
import os
import cv2
import numpy as np
import string
import random

classifications = string.ascii_lowercase + string.digits

model = None
pwd = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(pwd, 'model', 'model.json')) as model_json:
    model = model_from_json(model_json.read())

model.load_weights(os.path.join(pwd, 'model', 'weights.h5'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

def predict(filepath):
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = image/255.0
    else:
        print("Image not detected.")
        return
    result = np.array(model.predict(image[np.newaxis, :, :, np.newaxis]))
    answer = np.reshape(result, (5, 36)) # turn the result into a one-hot encoded vector.

    letter_index = []
    for vector in answer:
        letter_index.append(np.argmax(vector))

    captcha = str()
    for letter in letter_index:
        captcha += classifications[letter]
    return captcha
for filename in random.sample(os.listdir(os.path.join(pwd, 'output')), 10):
    prediction = predict(os.path.join(pwd, 'output', filename))
    actual = os.path.splitext(filename)[0]

    print("Read file %s:\nExpected: %s, Prediction: %s" % (filename, actual, prediction))