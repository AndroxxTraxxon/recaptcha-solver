import pandas as pd

import os

from keras import layers, callbacks, optimizers
from keras.models import Model, load_model

import cv2
import string
import numpy as np
import random

pwd = os.path.dirname(os.path.realpath(__file__))
classifications = string.ascii_lowercase + string.digits
max_files = 100000
num_classes = len(classifications)
image_shape = (60, 160, 1) # size of the images: 60 tall, 160 wide
epochs = 60

print("Searching for %d symbols... " % num_classes)

def build_model():
    image = layers.Input(shape=image_shape)
    conv_layers = [image]
    for filter_depth in (16, 32,):
        conv_layers.append(layers.Conv2D(filter_depth, (3,3), padding='same', activation='relu')(conv_layers[-1]))
        conv_layers.append(layers.MaxPooling2D(padding='same')(conv_layers[-1]))
    norm = layers.BatchNormalization()(conv_layers[-2])
    pool_d= layers.MaxPooling2D(padding='same')(norm)

    flattened = layers.Flatten()(pool_d)
    outputs = list()
    for i in range(5): 
        branch = [layers.Dense(num_classes * 4, activation='relu')(flattened)]
        branch.append(layers.Dense(num_classes * 2, activation='relu')(branch[-1]))
        branch.append(layers.Dropout(0.2)(branch[-1]))
        branch.append(layers.Dense((2 * num_classes) // 3, activation='sigmoid')(branch[-1]))
        branch.append(layers.Dense(num_classes, activation='sigmoid')(branch[-1]))
        outputs.append(branch[-1])

    model = Model(image, outputs)
    optimizer = optimizers.Adam(0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    return model

def preprocess_data():
    global max_files
    input_path = './output'
    dir_files = os.listdir(input_path)
    num_samples = min(max_files, len(dir_files))
    print("Processing %d of %d files in directory \n%s"%(num_samples, len(dir_files), input_path))
    samples = np.zeros((num_samples, *image_shape))
    labels = np.zeros((5, num_samples, num_classes))

    for i, name in enumerate(random.sample(dir_files, num_samples)):
        try:
            image = cv2.imread(os.path.join(input_path, name), cv2.IMREAD_GRAYSCALE)
            target = os.path.splitext(name)[0] # all the files are .jpg or .png. stripping the extension leaves the intended value
            if not target:
                print(name)
            if len(target) < 6:
                # scaling the image. 
                image = image/255.0
                # turning into a vector
                image = np.reshape(image, image_shape)
                # define the targets using one hot encoding
                targets = np.zeros((5, num_classes))
                for t, char in enumerate(target):
                    index = classifications.find(char)
                    targets[t, index] = 1
                samples[i] = image
                labels[:, i] = targets
        except:
            print("Finished processing %d files: " % i)
            print("Error processing file %s" % name)
            exit(0)
        

    
    return samples, labels

print("Preprocessing Data")
images, labels = preprocess_data()
train_count = (9*len(images))//10
print("Splitting training and test data sets.")
train_images, train_labels = images[:train_count], labels[:, :train_count]
test_images , test_labels = images[train_count:], labels[:, train_count:]
print("Building model")
model = build_model()

# model.summary()
try:
    input('Press Enter to continue...')
except KeyboardInterrupt:
    print("\nExiting...")
    exit()
model.fit(
    train_images, 
    [train_labels[0], train_labels[1], train_labels[2], train_labels[3], train_labels[4]], 
    batch_size=32, epochs=epochs,verbose=1, validation_split=0.1
)

scores = model.evaluate(test_images, [test_labels[0], test_labels[1], test_labels[2], test_labels[3], test_labels[4]], verbose=1)

print('Test Loss and accuracy: %s' % str(scores))

with open(os.path.join(pwd, 'model', 'model.json'), 'w+') as model_file:
    model_file.write(model.to_json())

with open(os.path.join(pwd, 'model', 'weights.h5'), 'wb+') as weights_file:
    model.save(weights_file)





