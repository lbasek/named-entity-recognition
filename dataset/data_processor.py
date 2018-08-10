from keras_preprocessing.sequence import pad_sequences


def numericalize(vocab, examples, pad_token, maxlen=None):
    data = [[vocab.stoi[word] for word in example] for example in examples]
    maxlen = maxlen if maxlen is not None else max(map(lambda e: len(e), data))
    return pad_sequences(data, padding='post', value=vocab.stoi[pad_token], maxlen=maxlen)
