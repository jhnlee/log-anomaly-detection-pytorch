import pickle
import os
from collections import Counter


class Vocab:
    def __init__(self, vocab_path):
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.vocab_path = vocab_path
        self.data = None
        self.vocab = None
        if os.path.isfile(vocab_path):
            self.load_vocab()

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data = list(data.values())

    def create_vocab(self):
        assert self.data is not None
        ctr = Counter()
        for d in self.data:
            ctr.update(d)
        vocab = list(ctr.keys())
        self.vocab = self.special_tokens + vocab
        self.save_vocab()

    def load_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = []
            for line in f:
                try:
                    vocab.append(int(line.strip()))
                except:
                    vocab.append(line.strip())
        self.vocab = vocab

    def save_vocab(self):
        assert self.vocab is not None
        with open(self.vocab_path, 'w') as f:
            for word in self.vocab:
                f.write(str(word) + '\n')

