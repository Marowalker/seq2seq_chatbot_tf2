import nltk
from collections import defaultdict
import os
import tensorflow as tf
import pickle
import constants


def daily_conversations(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        conversations = [line.split(' __eou__')[:-1] for line in f.readlines()]
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


def make_vocab(filein, fileout):
    if os.path.exists(fileout):
        with open(fileout, 'rb') as f:
            vocab = pickle.load(f)

    else:
        temp = []
        convo = daily_conversations(filein)
        for c in convo:
            for sent in c:
                words = nltk.word_tokenize(sent)
                for w in words:
                    temp.append(w.lower())
        all_words = sorted(set(temp))
        all_words.append(constants.START)
        all_words.append(constants.END)
        all_words.append(constants.UNK)
        vocab = defaultdict()
        for (idx, word) in enumerate(all_words):
            vocab[word] = idx + 1
        with open(fileout, 'wb') as f:
            pickle.dump(vocab, f)

    return vocab


def process_data(filename, vocab, max_length=constants.MAX_LENGTH):
    convo = daily_conversations(filename)
    questions, answers = conversation_to_qa(convo)

    start = [constants.START]
    end = [constants.END]

    inputs = []
    outputs = []

    for (q, a) in zip(questions, answers):
        q_tokens = nltk.word_tokenize(q)
        a_tokens = nltk.word_tokenize(a)
        q_tokens = start + q_tokens + end
        a_tokens = start + a_tokens + end

        q_idx = []
        a_idx = []

        for wq in q_tokens:
            if wq in vocab:
                q_idx.append(vocab[wq])
            else:
                q_idx.append(vocab[constants.UNK])

        for wa in a_tokens:
            if wa in vocab:
                a_idx.append(vocab[wa])
            else:
                a_idx.append(vocab[constants.UNK])

        inputs.append(q_idx)
        outputs.append(a_idx)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length, padding='post')
    padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, maxlen=max_length, padding='post')

    return padded_inputs, padded_outputs


def get_dataset(data_file, vocab_dict, out_file, max_length=constants.MAX_LENGTH):
    inputs, outputs = process_data(data_file, vocab_dict, max_length)
    dataset = {
        'inputs': inputs,
        'outputs': outputs
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
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
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
