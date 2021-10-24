import nltk
from collections import defaultdict
import os
import tensorflow as tf
import pickle
import constants
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def daily_conversations(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        conversations = [line.lower().split(' __eou__')[:-1] for line in f.readlines()]
        return conversations


def conversation_to_qa(conversation):
    questions = []
    answers = []
    for c in conversation:
        length = len(c)
        for i in range(length - 1):
            q = c[i]
            a = c[i + 1]
            questions.append(q)
            answers.append(a)
    return questions, answers


def make_tokenizer(filein):
    convo = daily_conversations(filein)
    sentence_list = []

    for c in convo:
        for s in c:
            s = constants.START + ' ' + s + ' ' + constants.END
            sentence_list.append(s)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence_list)

    return tokenizer


def preprocessing_keras(filename, tokenizer, max_length=constants.MAX_LENGTH):
    convo = daily_conversations(filename)
    questions, answers = conversation_to_qa(convo)

    new_questions = [constants.START + ' ' + q + ' ' + constants.END for q in questions]
    new_answers = [constants.START + ' ' + a + ' ' + constants.END for a in answers]

    inputs = tokenizer.texts_to_sequences(new_questions)
    outputs = tokenizer.texts_to_sequences(new_answers)

    padded_inputs = pad_sequences(inputs, maxlen=max_length, padding='post')
    padded_outputs = pad_sequences(outputs, maxlen=max_length, padding='post')

    return padded_inputs, padded_outputs


def get_dataset(data_file, out_file, tokenizer, max_length=constants.MAX_LENGTH):
    # inputs, outputs = process_data(data_file, vocab_dict, max_length)
    inputs, outputs = preprocessing_keras(data_file, tokenizer, max_length)
    dataset = {
        'inputs': inputs,
        'outputs': outputs,
    }
    with open(out_file, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def get_tf_dataset(inputs, outputs):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))
    dataset = dataset.shuffle(constants.BUFFER_SIZE)
    dataset = dataset.batch(constants.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    return dataset


def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def process_single(sentence, vocab):
    start = [constants.START]
    end = [constants.END]
    tokens = start + nltk.word_tokenize(sentence) + end
    res = []
    for w in tokens:
        if w in vocab:
            res.append(vocab[w])
        else:
            res.append(vocab[constants.UNK])
    return res

# vocab_words = make_vocab(filein=constants.DATA_FULL, fileout=constants.VOCAB)
# train = get_tf_dataset(data_file=constants.DATA_TRAIN, vocab_dict=vocab_words)
# print(train)
