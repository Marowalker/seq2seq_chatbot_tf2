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


def persona_conversations(filename):
    conversations = []
    temp = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '1' and temp:
                if '__SILENCE__' in temp:
                    conversations.append(temp[1:])
                else:
                    conversations.append(temp)
                temp = []
            temp += [l.strip() for l in line[2:].split('\t')]
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


def make_tokenizer(filein, persona_1, persona_2):
    convo = daily_conversations(filein)
    convo_p1 = persona_conversations(persona_1)
    convo_p2 = persona_conversations(persona_2)
    convo = convo + convo_p1 + convo_p2
    sentence_list = []

    for c in convo:
        for s in c:
            s = constants.START + ' ' + s + ' ' + constants.END
            sentence_list.append(s)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence_list)

    return tokenizer


def preprocessing_test(filename):
    convo = daily_conversations(filename)
    questions, answers = conversation_to_qa(convo)
    return questions, answers


def preprocessing_keras(daily_file, persona_file, tokenizer, max_length=constants.MAX_LENGTH, samples=None):
    convo = daily_conversations(daily_file)
    convo_p = persona_conversations(persona_file)
    convo = convo + convo_p
    questions, answers = conversation_to_qa(convo)

    new_questions = [constants.START + ' ' + q + ' ' + constants.END for q in questions]
    new_answers = [constants.START + ' ' + a + ' ' + constants.END for a in answers]

    if samples is not None:
        return questions[:samples], answers[:samples]

    inputs = tokenizer.texts_to_sequences(new_questions)
    outputs = tokenizer.texts_to_sequences(new_answers)

    padded_inputs = pad_sequences(inputs, maxlen=max_length, padding='post')
    padded_outputs = pad_sequences(outputs, maxlen=max_length, padding='post')

    return padded_inputs, padded_outputs


def get_dataset(daily_file, persona_file, out_file, tokenizer, max_length=constants.MAX_LENGTH):
    # inputs, outputs = process_data(data_file, vocab_dict, max_length)
    inputs, outputs = preprocessing_keras(daily_file, persona_file, tokenizer, max_length)
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

