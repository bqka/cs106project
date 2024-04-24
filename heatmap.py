import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from modelutils import *
from train import get_dataset
from keras.models import load_model

model = load_model('model/training_03.h5')

test_data, test_labels, _, _ = get_dataset()
true_labels = np.argmax(test_labels, axis=1)


predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()