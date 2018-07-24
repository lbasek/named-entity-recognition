import matplotlib.pyplot as plt

from embedding.glove import get_pretrained_glove
from root.dataset.api import load_dataset
from root.model import NeuralNetwork
from root.test_model import test_model

text_vocab, labels_vocab, pos_vocab, chunk_vocab, train, val, test = load_dataset()

num_words = len(text_vocab.itos)
num_entities = len(labels_vocab.itos)
num_pos = len(pos_vocab.itos)
num_chunk = len(chunk_vocab.itos)

nn = NeuralNetwork(num_words, num_entities, num_pos, num_chunk, train, test, val)

model, history = nn.train(epochs=5, embedding=get_pretrained_glove(num_words, text_vocab))

print(history.history.keys())

test_model(test, text_vocab, labels_vocab)

# Plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('../results/model_accuracy.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('../results/model_loss.png', dpi=200, format='png', bbox_inches='tight')
plt.close()
