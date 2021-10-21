import constants
import tensorflow as tf
from data_utils import *
from models.model import TransformerModel


tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE


def main():
    vocab = make_vocab(constants.DATA_FULL, constants.VOCAB)

    if constants.IS_REBUILD == 1:
        print("Buiding dataset objects...")
        train = get_dataset(constants.DATA_TRAIN, vocab, constants.PICKLE + 'train.pkl')
        dev = get_dataset(constants.DATA_DEV, vocab, constants.PICKLE + 'dev.pkl')
        # test = get_dataset(constants.DATA_TEST, vocab, constants.PICKLE + 'test.pkl')

    else:
        print("Loading dataset objects:...")
        train = load_dataset(constants.PICKLE + 'train.pkl')
        dev = load_dataset(constants.PICKLE + 'dev.pkl')

    print("Number of training samples:", len(train['inputs']))
    print("Number of validation samples:", len(dev['inputs']))

    train = get_tf_dataset(train['inputs'], train['outputs'])
    dev = get_tf_dataset(dev['inputs'], dev['outputs'])

    with tf.device('/device:GPU:0'):
        chatbot_model = TransformerModel(vocab, constants.NUM_LAYERS, constants.UNITS, constants.D_MODEL,
                                         constants.NUM_HEADS, constants.DROPOUT, constants.TRAINED_MODELS)
        chatbot_model.train(train, dev)

        # chatbot_model.predict('hello')


if __name__ == '__main__':
    main()
