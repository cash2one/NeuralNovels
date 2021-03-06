from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
from progressbar import ProgressBar
import os, random, sys, argparse, operator, gc
from time import sleep

from book_utils import *
from embedding_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # filter out INFO

model_file_name = 'checkpoints/1024_512_batchnorm.h5'

file_names = get_file_names_written_by('George Alfred Henty')
text = ''.join(get_files_contents(file_names))
print('text length', len(text))

print('Tokenizing...')
words, word_index, idx_to_word = tokenize_words(text, 20000)

# TODO: probably have a more random way of doing the train/val split
#val_split = 0.05
#train_words = words[:int((1 - val_split) * len(words))]
#val_words = words[int(val_split * len(words)):]

maxlen = 30
step = 3
batch_size = 1200
# Derived using formula for length of range at stackoverflow.com/questions/31839032
#total_train_samples = (len(train_words) - maxlen - 1) // step + 1
#total_val_samples = (len(val_words) - maxlen - 1) // step + 1
total_samples = (len(words) - maxlen - 1) // step + 1 
def get_chunk(data):
    # cut the text in semi-redundant sequences of maxlen words
    # I derived this using the formula for the length of a range and wolfram alpha. It should work
    words_per_batch = (batch_size - 1) * step + maxlen + 1
    words_order = list(range(len(data) // words_per_batch - 1))

    while True:
        random.shuffle(words_order)
        for words_idx in words_order:
            words_batch = data[words_idx * words_per_batch : (words_idx + 1) * words_per_batch]
            sentences = []
            next_words = []

            for i in range(0, words_per_batch - maxlen, step):
                sentences.append(words_batch[i: i + maxlen])
                next_words.append(words_batch[i + maxlen])

            x = np.array(sentences)
            y = to_categorical(next_words, num_classes=len(word_index))

            yield shuffle(x, y)

def build_model(load_weights):
    print('Build model...')
    if load_weights:
        return load_model(model_file_name)
    else:
        model = Sequential([
                    get_embedding_layer(word_index, maxlen, trainable=True),
                    BatchNormalization(),
                    LSTM(1024, dropout=0.5, return_sequences=True),
                    BatchNormalization(),
                    LSTM(512, dropout=0.5),
                    BatchNormalization(),
                    Dense(len(word_index), activation='softmax')
            ])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.summary()
        return model

def train(n_iter, n_words, beam_width, load_checkpoint):
    model = build_model(load_checkpoint)
    for iteration in range(1, n_iter + 1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit_generator(get_chunk(words),
                            #(total_train_samples // batch_size) // 4,
                            total_samples // batch_size,
                            epochs=1,
                            #validation_data=get_chunk(val_words),
                            #validation_steps=total_val_samples // batch_size
                            )

        print('Saving model...')
        model.save(model_file_name)

        for diversity in [1.4, 1.6, 1.8, 2.0]:
            print()
            beam_search(model, n_words, beam_width, diversity, sys.stdout)
            print()

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
    init_sentence = np.concatenate((init_sentence, np.zeros(n_words))).astype(np.int)

    beams = [{ 'sentence': init_sentence, 'proba': 1.0 }]

    print('Generating with beam search...')
    bar = ProgressBar()
    for i in bar(range(n_words)):
        new_beams = []

        x_pred = np.array([beam['sentence'][i:i + maxlen] for beam in beams])
        all_preds = model.predict(x_pred, verbose=0, batch_size=batch_size)
        for preds, beam in zip(all_preds, beams):
            beam['preds'] = preds

        for idx, beam in enumerate(beams):
            sentence = beam['sentence']
            preds = scale_prediction(beam['preds'], diversity)

            best_idxs = np.random.choice(
                            len(preds),
                            size=min(np.count_nonzero(preds > 0), beam_width),
                            replace=False,
                            p=preds
                        )

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

parser = argparse.ArgumentParser(description='train/generate text with word rnn')
parser.add_argument('--mode', type=str, default='train', help='Either "train" or "generate"')
parser.add_argument('--iter', type=int, default=3, help='Number of training iterations')
parser.add_argument('--words', type=int, default=80, help='Number of words for  beam search')
parser.add_argument('--diversity', type=float, default=1.6, help='Temperature for beam search')
parser.add_argument('--beam_width', type=int, default=30, help='Beam width for beam search')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true', help='Load the model from a checkpoint file')
parser.set_defaults(load_checkpoint=False)

FLAGS = parser.parse_args()

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        trained_model = train(FLAGS.iter, FLAGS.words, FLAGS.beam_width, FLAGS.load_checkpoint)
        beam_search(trained_model, 1000, FLAGS.beam_width, FLAGS.diversity, open('./generated_words.md', 'w'))
    elif FLAGS.mode == 'generate':
        beam_search(build_model(True), FLAGS.words, FLAGS.beam_width, FLAGS.diversity, open('./generated_words.md', 'w'))
    else:
        print('Unrecognized mode', FLAGS.mode)
        raise TypeError

