'''
At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import GRU
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import argparse

from book_utils import *

file_names = get_file_names_written_by('George Alfred Henty')
text = ''.join(get_files_contents(file_names))[:4000000]
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model
print('Build model...')
model = Sequential()
model.add(GRU(256, input_shape=(maxlen, len(chars)), return_sequences=True, dropout=0.25, recurrent_dropout=0.1))
model.add(GRU(128, dropout=0.3, recurrent_dropout=0.1))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train(n_iter):
    # train the model, output generated text after each iteration
    for iteration in range(1, n_iter):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x, y,
                  batch_size=256,
                  epochs=2,
                  validation_split=0.05)
    
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            print()
    
    print('Saving model...')
    model.save('gru_char_rnn.h5')

def generate(n_chars, diversity=0.6, stream=sys.stdout):
    model.load_weights('gru_char_rnn.h5')
    start_index = random.randint(0, len(text) - maxlen - 1)
    
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    stream.write(generated)
    
    for i in range(n_chars):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
    
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
    
        generated += next_char
        sentence = sentence[1:] + next_char
    
        stream.write(next_char)
        stream.flush()

parser = argparse.ArgumentParser(description='train/generate text with GRU char rnn')
parser.add_argument('--mode', type=str, default='train', help='Either "train" or "generate"')
parser.add_argument('--iter', type=int, default=60, help='Number of training iterations')
parser.add_argument('--chars', type=int, default=1000, help='Number of characters to generate')

FLAGS = parser.parse_args()

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train(FLAGS.iter)
    elif FLAGS.mode == 'generate':
        generate(FLAGS.chars, stream=open('./generated.txt', 'w'))
    else:
        print('Unrecognized mode', FLAGS.mode)
        raise TypeError

