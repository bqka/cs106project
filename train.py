import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from keras.utils import load_img
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from modelutils import *
from keras.callbacks import ModelCheckpoint

def create_dataframe(dir):
    image_paths = []
    labels = []

    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
    
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def get_dataset(TRAIN_DIR = 'images/train', TEST_DIR = 'images/test'):
    train = pd.DataFrame()
    train['image'], train['label'] = create_dataframe(TRAIN_DIR)

    test = pd.DataFrame()
    test['image'], test['label'] = create_dataframe(TEST_DIR)

    # train_features = extract_features(train['image'])
    # test_features = extract_features(test['image'])

    train_features = np.load('features/train.npy')
    test_features = np.load('features/test.npy')

    x_train = train_features/255.0
    x_test = test_features/255.0
    
    le = LabelEncoder()
    le.fit(train['label'])
    
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_dataset()

# model = create_model()
model = load('training_02')
model = compile(model)

model = train(model, x_train, y_train, x_test, y_test)

save(model, "training_03")
