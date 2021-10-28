import constants
from preprocessing.data_to_sequence import *
from models.self_transformer import TransformerModel
import numpy as np
from preprocessing.data_to_pandas import *
from models.simple_seq2seq import SimpleSeq2Seq


tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE


def main_self_transformer():
    # vocab = make_vocab(constants.DATA_FULL, constants.VOCAB)
    tokenizer = make_tokenizer(constants.DATA_FULL)

    if constants.IS_REBUILD == 1:
        print("Buiding dataset objects...")
        train = get_dataset(constants.DATA_TRAIN, constants.PICKLE + 'train.pkl', tokenizer)
        dev = get_dataset(constants.DATA_DEV, constants.PICKLE + 'dev.pkl', tokenizer)
        # test = get_dataset(constants.DATA_TEST, constants.PICKLE + 'test.pkl', tokenizer)

    else:
        print("Loading dataset objects:...")
        train = load_dataset(constants.PICKLE + 'train.pkl')
        dev = load_dataset(constants.PICKLE + 'dev.pkl')
        # test = load_dataset(constants.PICKLE + 'test.pkl')

    print("Number of training samples:", len(train['inputs']))
    print("Number of validation samples:", len(dev['inputs']))

    train = get_tf_dataset(train['inputs'], train['outputs'])
    dev = get_tf_dataset(dev['inputs'], dev['outputs'])

    test_questions, test_answers = preprocessing_test(constants.DATA_TEST)

    with tf.device('/device:GPU:0'):
        chatbot_model = TransformerModel(tokenizer, constants.NUM_LAYERS, constants.UNITS, constants.D_MODEL,
                                         constants.NUM_HEADS, constants.DROPOUT,
                                         constants.TRAINED_MODELS + 'self_transformer/')
        # chatbot_model.train(train, dev)

        # evaluate test set
        chatbot_model.evaluate_coherence(test_questions)
        # chatbot_model.evaluate_bleu(test_questions, test_answers)
        # chatbot_model.evaluate_rouge(test_questions, test_answers)

        # sentence = ['hello there', 'nice to meet you']
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
    main_self_transformer()
    # main_seq2seq()
