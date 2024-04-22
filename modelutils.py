from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
from keras.utils import load_img
import numpy as np

def load(json_file='emotiondetector.json', h5_file='emotiondetector.h5'):
    json_file = open(json_file, 'r')
    data = json_file.read()
    model = model_from_json(data)
    model.load_weights(h5_file)
    
    return model

def create():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    # fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # output layer
    model.add(Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    
    return model
    
def train(model, x_train, y_train, x_test, y_test):
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
    
    return train

def ef(image):
    img = load_img(image, color_mode="grayscale")
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    
    return feature/255.0

def predict(model, image):
    label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    image = ef(image)
    pred = model.predict(image)
    pred_label = label[pred.argmax()]
    
    return pred_label