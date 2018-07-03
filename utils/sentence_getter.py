class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False

        self.sentences = list()
        sentence = list()
        for index, row in data.iterrows():
            if row['Word'] == '.' and row['POS'] == '.' and row['Chunk'] == 'O' and row['Tag'] == 'O':
                if len(sentence) == 0:
                    continue
                self.sentences.append(sentence)
                sentence = list()
            else:
                sentence.append((row['Word'], row['POS'], row['Chunk'], row['Tag']))

    def get_next(self):
        try:
            s = self.sentences[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
