import matplotlib.pyplot as plt

from root.model import NeuralNetwork
from root.preprocessing import Preprocessing, Dataset

TRAIN = '../dataset/csv/train.csv'
TEST = '../dataset/csv/test.csv'
VALIDATION = '../dataset/csv/valid.csv'

preprocessing = Preprocessing(TRAIN, TEST, VALIDATION)

X_train, Y_train, num_entities, num_words = preprocessing.create_input(Dataset.train)
X_validation, Y_validation, _, _ = preprocessing.create_input(Dataset.validation)

nn = NeuralNetwork(num_words, num_entities, X_train, Y_train, X_validation, Y_validation)
model, history = nn.train()

print(history.history.keys())

# Plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
