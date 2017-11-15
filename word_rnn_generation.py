from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import GRU
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from progressbar import ProgressBar
import random, sys, argparse, operator

from book_utils import *
from embedding_utils import *

file_names = get_file_names_written_by('George Alfred Henty')
text = ''.join(get_files_contents(file_names))[:2000000]

print('Tokenizing...')
words, word_index = tokenize_words(text, 12000)
idx_to_word = {v: k for k, v in word_index.items()} 

# cut the text in semi-redundant sequences of maxlen words
maxlen = 20
step = 3
sentences = []
next_words = []
for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_words.append(words[i + maxlen])
print('nb sequences:', len(sentences))
x = np.array(sentences)
y = to_categorical(next_words)
print(x[0])

def build_model(load_weights):
    # build the model
    print('Build model...')
    if load_weights:
        return load_model('gru_word_rnn.h5')
    else:
        model = Sequential()
        model.add(get_embedding_layer(word_index, maxlen))
        model.add(GRU(128, dropout=0.5, recurrent_dropout=0.25, return_sequences=True))
        model.add(GRU(128, dropout=0.5, recurrent_dropout=0.25, return_sequences=True))
        model.add(GRU(128, dropout=0.5, recurrent_dropout=0.25))
        model.add(Dense(len(word_index), activation='softmax'))
        return model

def train(n_iter, n_words, beam_width):
    # train the model, output generated text after each iteration
    model = build_model(False)
    for iteration in range(1, n_iter):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x, y,
                  batch_size=512,
                  epochs=2,
                  validation_split=0.05)
    
        for diversity in [1.2, 1.4, 1.6, 1.8]:
            print()
            beam_search(model, n_words, beam_width, diversity, sys.stdout)
            print()
    
    print('Saving model...')
    model.save('gru_word_rnn.h5')
    return model

def scale_prediction(prediction, temperature):
    '''stolen from pender/chatbot-rnn chatbot.py'''
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return (scaled_prediction / np.sum(scaled_prediction)) # renormalize

def beam_search(model, n_words, beam_width, diversity, stream):
    print('----- diversity:', diversity)
    start_index = random.randint(0, len(words) - maxlen - 1)
    init_sentence = np.array(words[start_index: start_index + maxlen])
    init_sentence = np.concatenate((init_sentence, np.zeros(n_words)))

    beams = [{ 'sentence': init_sentence, 'proba': 1.0 }]

    print('Generating with beam search...')
    bar = ProgressBar()
    for i in bar(range(n_words)):
        new_beams = []

        x_pred = np.array([beam['sentence'][i:i + maxlen] for beam in beams])
        all_preds = model.predict(x_pred, verbose=0, batch_size=256)
        for preds, beam in zip(all_preds, beams):
            beam['preds'] = preds

        for idx, beam in enumerate(beams):
            sentence = beam['sentence']
            preds = scale_prediction(beam['preds'], diversity)

            best_idxs = np.random.choice(len(preds), size=beam_width, replace=False, p=preds)

            for pred_idx in best_idxs:
                new_sentence = np.copy(sentence)
                new_sentence[maxlen + i] = pred_idx
                new_beams.append({
                                    'sentence': new_sentence,
                                    'proba': beam['proba'] * preds[pred_idx]
                                })

        beams = list(reversed(sorted(new_beams, key=lambda beam: beam['proba'])))
        beams = beams[:beam_width]
        sum_probs = sum([beam['proba'] for beam in beams])
        for beam in beams: beam['proba'] /= sum_probs # avoid going to zero

    stream.write(detokenize(beams[0]['sentence'], idx_to_word))

parser = argparse.ArgumentParser(description='train/generate text with GRU word rnn')
parser.add_argument('--mode', type=str, default='train', help='Either "train" or "generate"')
parser.add_argument('--iter', type=int, default=80, help='Number of training iterations')
parser.add_argument('--words', type=int, default=50, help='Number of words for  beam search')
parser.add_argument('--diversity', type=float, default=1.2, help='Temperature for beam search')
parser.add_argument('--beam_width', type=int, default=30, help='Beam width for beam search')

FLAGS = parser.parse_args()

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        trained_model = train(FLAGS.iter, FLAGS.words, FLAGS.beam_width)
        beam_search(trained_model, 1000, FLAGS.beam_width, FLAGS.diversity, open('./generated_words.md', 'w'))
    elif FLAGS.mode == 'generate':
        beam_search(build_model(True), FLAGS.words, FLAGS.beam_width, FLAGS.diversity, open('./generated_words.md', 'w'))
    else:
        print('Unrecognized mode', FLAGS.mode)
        raise TypeError

