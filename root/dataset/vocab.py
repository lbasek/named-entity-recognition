from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from root.constants import UNK, UNK_POS, PAD


class Vocab(ABC):
    def __init__(self):
        self._stoi = dict()
        self._itos = []

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos

    @staticmethod
    @abstractmethod
    def build(sentences):
        raise NotImplementedError


class TextVocab(Vocab):
    @staticmethod
    def build(sentences, max_size=None):
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)

        max_size = len(counter) if max_size is None else max_size
        words_and_freqs = counter.most_common(max_size)

        vocab = TextVocab()
        vocab._itos = [UNK] + list(map(lambda t: t[0], words_and_freqs))
        vocab._stoi = defaultdict(lambda: 0)  # index of UNK token
        vocab.stoi.update({k: v for v, k in enumerate(vocab.itos)})

        return vocab


class LabelVocab(Vocab):
    @staticmethod
    def build(sentences):
        unique_labels = set()
        for labels in sentences:
            unique_labels.update(labels)

        vocab = LabelVocab()
        vocab._itos = list(sorted(unique_labels))
        vocab._stoi = defaultdict(lambda: 0)
        vocab.stoi.update({k: v for v, k in enumerate(vocab.itos)})

        return vocab


class PosVocab(Vocab):
    @staticmethod
    def build(sentences):
        unique_pos = set()
        for pos in sentences:
            unique_pos.update(pos)

        vocab = PosVocab()
        vocab._itos = [UNK_POS] + list(sorted(unique_pos))
        vocab._stoi = defaultdict(lambda: 0)
        vocab.stoi.update({k: v for v, k in enumerate(vocab.itos)})

        return vocab
