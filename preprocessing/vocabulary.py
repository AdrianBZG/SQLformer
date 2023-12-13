"""
Classes to handle vocabularies
"""
from collections import Counter
import random
import numpy as np
import torch
import logging

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("vocabulary")


class Vocabulary(object):
    def __init__(self, sentence_model):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "<sep>": 4}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>", 4: "<sep>"}
        self.idx = len(self.word2idx)
        self.vocab_cnt = None
        self.word2embedding = dict()

        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"
        self.UNK_TOKEN = "<unk>"
        self.SEP_TOKEN = "<sep>"
        self.SPECIAL_TOKENS = [self.PAD_TOKEN,
                               self.SOS_TOKEN,
                               self.EOS_TOKEN,
                               self.UNK_TOKEN,
                               self.SEP_TOKEN]

        self.sentence_model = sentence_model['model']
        self.tokenizer = sentence_model['tokenizer']

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def init_table_alias_tokens(self):
        for i in range(10):
            self.add_word(f"T{i}")

    def is_in_vocab(self, word):
        if word in self.word2idx:
            return True

        return False

    def to_idx(self, word):
        if word not in self.word2idx:
            return self.word2idx.get("<unk>")

        w_idx = self.word2idx.get(word)
        return w_idx

    def to_token(self, idx):
        if idx not in self.idx2word:
            return "<unk>"

        return self.idx2word.get(idx)

    def merge_vocabulary(self, vocab):
        for word in vocab.word2idx:
            self.add_word(word)

    def update_counter(self):
        self.vocab_cnt = Counter(list(self.word2idx.keys()))
        self.vocab_cnt = sorted(list(self.vocab_cnt.items()),
                                key=lambda x: - x[1])

    def get_counter(self):
        self.update_counter()
        return self.vocab_cnt

    def get_word_embedding(self, word, device="gpu"):
        if word not in self.word2embedding:
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            emb = self.sentence_model.encode([word],
                                             show_progress_bar=False,
                                             convert_to_tensor=True)
            self.word2embedding[word] = emb[0]

        if device == "cpu":
            return self.word2embedding[word].to("cpu")

        return self.word2embedding[word]

    def as_list(self):
        return list(self.word2idx.keys())

    def token_to_one_hot(self, token):
        w_idx = self.word2idx.get(token)
        one_hot = [0] * len(self.word2idx)
        one_hot[w_idx] = 1
        return one_hot

    def get_vocabulary_embedding(self):
        for tkinx, token in enumerate(self.SPECIAL_TOKENS):
            emb = self.sentence_model.encode([token],
                                             show_progress_bar=False)
            self.word2embedding[token] = emb[0]

        logger.info(f"Generating embeddings for {len(self.word2idx)} words")
        all_vocab_words = [word for word in self.word2idx if word not in self.SPECIAL_TOKENS]
        all_vocab_words_emb = self.sentence_model.encode(all_vocab_words,
                                                         show_progress_bar=False)
        for inx, word in enumerate(all_vocab_words):
            self.word2embedding[word] = all_vocab_words_emb[inx]

        logger.info(f"Finished embeddings for {len(self.word2embedding)} words")
