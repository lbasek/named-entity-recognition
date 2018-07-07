import matplotlib.pyplot as plt
from root.model import NeuralNetwork
from root.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

preprocessing = Preprocessing()

All_X_train, All_Y_train, num_entities, num_words = preprocessing.create_input()

X_rest, X_validation, y_rest, Y_validation = train_test_split(All_X_train, All_Y_train, test_size=0.2, train_size=0.8)

X_train, X_test, Y_train, Y_test = train_test_split(X_rest, y_rest, test_size=0.25, train_size=0.75)

nn = NeuralNetwork(num_words, num_entities, X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
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
