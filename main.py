import constants
from preprocessing.data_to_sequence import *
from models.self_transformer import TransformerModel
import numpy as np
from preprocessing.data_to_pandas import *
from models.simple_seq2seq import SimpleSeq2Seq


tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE


def main_self_transformer():
    vocab = make_vocab(constants.DATA_FULL, constants.VOCAB)

    if constants.IS_REBUILD == 1:
        print("Buiding dataset objects...")
        train = get_dataset(constants.DATA_TRAIN, vocab, constants.PICKLE + 'train.pkl')
        dev = get_dataset(constants.DATA_DEV, vocab, constants.PICKLE + 'dev.pkl')
        test = get_dataset(constants.DATA_TEST, vocab, constants.PICKLE + 'test.pkl')

    else:
        print("Loading dataset objects:...")
        train = load_dataset(constants.PICKLE + 'train.pkl')
        dev = load_dataset(constants.PICKLE + 'dev.pkl')
        test = load_dataset(constants.PICKLE + 'test.pkl')

    props = ['inputs', 'outputs']
    for prop in props:
        train[prop] = np.concatenate((train[prop], dev[prop]), axis=0)
        dev[prop] = test[prop]

    print("Number of training samples:", len(train['inputs']))
    print("Number of validation samples:", len(dev['inputs']))

    train = get_tf_dataset(train['inputs'], train['outputs'])
    dev = get_tf_dataset(dev['inputs'], dev['outputs'])

    with tf.device('/device:GPU:0'):
        chatbot_model = TransformerModel(vocab, constants.NUM_LAYERS, constants.UNITS, constants.D_MODEL,
                                         constants.NUM_HEADS, constants.DROPOUT,
                                         constants.TRAINED_MODELS + 'self_transformer/')
        chatbot_model.train(train, dev)

        # sentence = 'hello there'
        # sentence = process_single(sentence, vocab)
        # chatbot_model.predict(sentence)


def main_seq2seq():
    train = qa_to_pandas(constants.DATA_TRAIN, fileout=constants.DATA + 'train/train.csv')
    dev = qa_to_pandas(constants.DATA_DEV, fileout=constants.DATA + 'validation/validation.csv')
    test = qa_to_pandas(constants.DATA_TEST, fileout=constants.DATA + 'test/test.csv')

    model = SimpleSeq2Seq('bert', 'bert-base-uncased', 'bert-base-uncased', constants.EPOCHS,
                          constants.SEQ2SEQ + 'train/', constants.SEQ2SEQ + 'eval/')
    model.train(train, dev)

    model.evaluate(test)

    sentence = ['hello there']

    model.predict(sentence)


if __name__ == '__main__':
    # main_self_transformer()
    main_seq2seq()
