import os
import torch
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import warnings
warnings.filterwarnings("ignore")

CLS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
UNK_token = 2  # Unknown token
MAX_LENGTH = 10

def load_glove_embeddings(glove_file_path, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

class Lang:
    def __init__(self, name, glove_embeddings, embedding_dim=100):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.n_words = 3  # Start with 3 for CLS, EOS, and UNK
        self.embedding_dim = embedding_dim
        self.embeddings = []

        # Initialize with GloVe embeddings
        self.glove_embeddings = glove_embeddings

        # Add special tokens
        self.addSpecialToken("CLS")
        self.addSpecialToken("EOS")
        self.addSpecialToken("UNK")
    
    def addSpecialToken(self, token):
        self.word2index[token] = self.n_words - 3 + len(self.embeddings)  # Position according to the special token index
        self.index2word[self.n_words - 3 + len(self.embeddings)] = token
        if token in self.glove_embeddings:
            self.embeddings.append(self.glove_embeddings[token])
        else:
            self.embeddings.append(np.random.normal(size=(self.embedding_dim,)))

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            if word in self.glove_embeddings:
                self.embeddings.append(self.glove_embeddings[word])
            else:
                self.embeddings.append(self.glove_embeddings["UNK"])  # Use UNK embedding for unknown words
            self.n_words += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def get_embeddings_matrix(self):
        return torch.tensor(np.array(self.embeddings), dtype=torch.float32)

 
def translateDatasetEntries(dataset, lang):
    translated_texts = []
    labels = []

    for i in range(len(dataset)):
        text_tensor, label_tensor = dataset[i] 
        text_indices = text_tensor.numpy() 

        translated_text = lang.indicesToString(text_indices)
        translated_texts.append(translated_text)
        
        labels.append(label_tensor.item())

    return translated_texts, labels


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


class ToxicityDataset(Dataset):
    def __init__(self, filename, id_col, text_col, label_col, lang):
        self.data = pd.read_csv(filename, quoting=3)
        self.data[label_col] = self.data[label_col].apply(lambda x: -1 if x == '?' else x)
        self.data[text_col] = self.data[text_col].apply(lambda x: unicodeToAscii(x))
        self.data[text_col] = self.data[text_col].apply(lambda x: normalizeString(x))

        self.lang = lang
        self.texts = [self.encode_sentence(text) for text in self.data[text_col].values]
        self.labels = self.data[label_col].values

    def encode_sentence(self, sentence):
        encoded_sentence = []
        for word in sentence.split(' '):
            if word in self.lang.word2index:
                encoded_sentence.append(self.lang.word2index[word])
            else:
                encoded_sentence.append(UNK_token)
        return encoded_sentence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_encoded = self.texts[idx]
        label = self.labels[idx]
        text_encoded_with_eos = text_encoded + [EOS_token]
        return torch.tensor(text_encoded_with_eos, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def collate(batch):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (src_seq, tgt_seq):
          src_seq is of shape (src_seq_length,)
          tgt_seq is of shape (tgt_seq_length,)

    Returns:
      src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
          The sequences should be sorted by length in a decreasing order, that is src_seqs[:,0] should be
          the longest sequence, and src_seqs[:,-1] should be the shortest.
      src_seq_lengths: List of lengths of source sequences.
      tgt_seqs of shape (max_tgt_seq_length, batch_size): Tensor of padded target sequences.
    """
    batch_sorted = sorted(batch, key= lambda x: len(x[0]), reverse = True)
    texts, labels = zip(*batch_sorted)

    cls_tensor = torch.tensor([CLS_token])
    texts = [torch.cat((cls_tensor, tg)) for tg in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    src_mask = (texts_padded == 0)
    src_mask[:,0] = False

    labels = torch.tensor(labels, dtype=torch.float32)

    return texts_padded, src_mask, labels