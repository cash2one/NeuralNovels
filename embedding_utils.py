import os
from collections import Counter

import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer

def tokenize_words(text, vocab_size):
    open('/tmp/in.txt', 'w').write(text)
    os.system('~/Code/dl/datasets/stanford-parser-full-2017-06-09/tokenize.sh')
    tokens = open('/tmp/out.txt').read().split('\n')

    top_tokens = [tok for (tok, cnt) in Counter(tokens).most_common(vocab_size)]
    word_index = {tok: idx for (idx, tok) in enumerate(top_tokens)}
    unk_index = len(word_index)
    word_index['<UNK>'] = unk_index
    sequences = [word_index.get(tok, unk_index) for tok in tokens]

    print('Found {} unique tokens.'.format(len(word_index)))
    return (sequences, word_index)

def detokenize(words, idx_to_word):
    word_list = [idx_to_word[idx] for idx in words]
    open('/tmp/in.txt', 'w').write('\n'.join(word_list))
    os.system('~/Code/dl/datasets/stanford-parser-full-2017-06-09/detokenize.sh')
    return open('/tmp/out.txt').read()

def get_embedding_layer(word_index, input_length):
    EMBEDDING_DIM = 300
    embeddings_index = {}
    f = open(os.path.expanduser('~/Code/dl/datasets/glove.42B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word.lower())
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    return embedding_layer

