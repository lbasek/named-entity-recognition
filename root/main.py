import matplotlib.pyplot as plt
from datetime import datetime
from utils.serialization import save_object

from embedding.glove import get_pretrained_glove
from root.dataset.api import load_dataset
from root.model import NeuralNetwork
from root.test_model import test_model

text_vocab, labels_vocab, pos_vocab, train, val, test = load_dataset()

num_words = len(text_vocab.itos)
num_entities = len(labels_vocab.itos)
num_pos = len(pos_vocab.itos)

# save vocabulary
save_path = '../models/' + datetime.now().strftime("%Y-%m-%d-%H:%M") + '/'
save_object(text_vocab, save_path + 'text_vocab')
save_object(labels_vocab, save_path + 'labels_vocab')

nn = NeuralNetwork(save_path, num_words, num_entities, train.X, train.y, val.X, val.y, test.X, test.y)

model, history = nn.train(epochs=1, embedding=get_pretrained_glove(num_words, text_vocab))

print(history.history.keys())

test_model(save_path, test.X, test.y, text_vocab, labels_vocab)

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
