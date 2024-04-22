import train
import modelutils

x_train, y_train, x_test, y_test = train.get_dataset()

model = modelutils.load()
image = 'images/test/surprise/78.jpg'
print(modelutils.predict(model, image))