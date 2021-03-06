from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, BatchNormalization, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import numpy as np
from progressbar import ProgressBar
import random, sys, argparse, operator, gc
from time import sleep

from book_utils import *
from embedding_utils import *

file_names = get_file_names_written_by('George Alfred Henty')
text = ''.join(get_files_contents(file_names))

max_word_len = 50
words, char_index, idx_to_char = tokenize_words_to_chars(text, max_word_length=max_word_len)

# TODO: probably have a more random way of doing the train/val split
val_split = 0.05
train_words = words[:int((1 - val_split) * len(words))]
val_words = words[int(val_split * len(words)):]

maxlen = 30
step = 3
batch_size = 512
# Derived using formula for length of range at stackoverflow.com/questions/31839032
total_train_samples = (len(train_words) - maxlen - 1) // step + 1
total_val_samples = (len(val_words) - maxlen - 1) // step + 1
def get_chunk(data):
    # cut the text in semi-redundant sequences of maxlen words
    words_idx = 0

    # I derived this using the formula for the length of a range and wolfram alpha. It should work
    words_per_batch = (batch_size - 1) * step + maxlen + 1
    while True:
        if (words_idx + 1) * words_per_batch >= len(data): words_idx = 0
        words_batch = data[words_idx * words_per_batch : (words_idx + 1) * words_per_batch]
        sentences = []
        next_chars = []

        for i in range(0, words_per_batch - maxlen, step):
            sentences.append(words_batch[i: i + maxlen])
            next_chars.append(words_batch[i + maxlen])
        import pdb; pdb.set_trace()

        x = np.array([to_categorical(sentence, num_classes=len(char_index)) for sentence in sentences])
        y = to_categorical(next_chars, num_classes=len(char_index))

        yield (x, y)

        words_idx += 1

def build_model(load_weights):
    # build the model
    print('Build model...')
    if load_weights:
        return load_model('gru_char_rnn.h5')
    else:
        model = Sequential([
                    TimeDistributed(Conv1D(300, 5, activation='relu'), input_shape=(maxlen, max_word_len, len(char_index))),
                    TimeDistributed(MaxPooling1D()),
                    TimeDistributed(Flatten()),
                    LSTM(1024, return_sequences=True),
                    LSTM(1024),
                    Dense(len(char_index), activation='softmax')
            ])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.summary()
        return model

def train(n_iter, n_chars, beam_width):
    # train the model, output generated text after each iteration
    model = build_model(False)
    for iteration in range(1, n_iter + 1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit_generator(get_chunk(train_words),
                            total_train_samples // batch_size,
                            epochs=1,
                            validation_data=get_chunk(val_words),
                            validation_steps=total_val_samples // batch_size)

        for diversity in [1.2, 1.4, 1.6, 1.8]:
            print()
            beam_search(model, n_chars, beam_width, diversity, sys.stdout)
            print()
    
    print('Saving model...')
    model.save('gru_char_rnn.h5')
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

    stream.write(detokenize(beams[0]['sentence'], idx_to_char))

parser = argparse.ArgumentParser(description='train/generate text with char rnn')
parser.add_argument('--mode', type=str, default='train', help='Either "train" or "generate"')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations')
parser.add_argument('--words', type=int, default=80, help='Number of words for  beam search')
parser.add_argument('--diversity', type=float, default=1.6, help='Temperature for beam search')
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

