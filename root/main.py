import matplotlib.pyplot as plt
from root.model import NeuralNetwork
from root.dataset.api import load_dataset
from root.test_model import test_model

text_vocab, labels_vocab, train, val, test = load_dataset()

num_words = len(text_vocab.itos)
num_entities = len(labels_vocab.itos)

nn = NeuralNetwork(num_words, num_entities, train.X, train.y, val.X, val.y, test.X, test.y)
model, history = nn.train()

print(history.history.keys())

test_model(test.X, test.y, text_vocab, labels_vocab)

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
