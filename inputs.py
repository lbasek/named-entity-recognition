from keras import Input
from keras.layers import Embedding, Dropout, LSTM, TimeDistributed, SpatialDropout1D, concatenate
from embedding.glove import get_pretrained_glove
from constants import MAX_LEN, MAX_LEN_CHAR


def inputs_factory(args, vocabs):
    inputs = []
    input_layers = []

    for key, func in inputs_map.items():
        if key in args.inputs:
            input, input_layer = func(args, vocabs)
            inputs.append(input)
            input_layers.append(input_layer)

    # Concatenate inputs (if there are multiple)
    if len(inputs) > 1:
        model_input = concatenate(input_layers, axis=2)
    else:
        model_input = input_layers[0]

    model_input = SpatialDropout1D(0.3)(model_input)

    return inputs, model_input


def words_input(args, vocabs):
    num_words = len(vocabs.words.itos)
    txt_input = Input(shape=(None,), name='txt_input')
    if args.embeddings_type == 'glove':
        txt_embed = Embedding(input_dim=num_words, output_dim=MAX_LEN, input_length=None,
                              name='txt_embedding', trainable=args.embeddings_trainable,
                              weights=([get_pretrained_glove(num_words, vocabs.words)]))(txt_input)
    else:
        txt_embed = Embedding(input_dim=num_words, output_dim=MAX_LEN, input_length=None,
                              name='txt_embedding', trainable=args.embeddings_trainable)(txt_input)

    txt_drpot = Dropout(0.1, name='txt_dropout')(txt_embed)

    return txt_input, txt_drpot


def pos_input(args, vocabs):
    pos_input = Input(shape=(None,), name='pos_input')
    pos_embed = Embedding(input_dim=len(vocabs.pos.itos), output_dim=MAX_LEN, input_length=None, name='pos_embedding')(
        pos_input)
    pos_drpot = Dropout(0.1, name='pos_dropout')(pos_embed)
    return pos_input, pos_drpot


def chars_input(args, vocabs):
    char_in = Input(shape=(None, MAX_LEN_CHAR,), name="char_input")
    emb_char = TimeDistributed(Embedding(input_dim=len(vocabs.chars.itos), output_dim=MAX_LEN_CHAR, input_length=None))\
        (char_in)
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)
    return char_in, char_enc


inputs_map = {
    'words': words_input,
    'pos': pos_input,
    'chars': chars_input
}
