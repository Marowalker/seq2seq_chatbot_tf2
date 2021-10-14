import constants
import tensorflow as tf
from data_utils import *
from models.model import TransformerModel


def main():
    vocab = make_vocab(constants.DATA_FULL, constants.VOCAB)
    print("Buiding dataset objects...")
    train = get_tf_dataset(constants.DATA_TRAIN, vocab)
    dev = get_tf_dataset(constants.DATA_DEV, vocab)
    test = get_tf_dataset(constants.DATA_TEST, vocab)

    print("Done building datasets.\n")

    with tf.device('/device:GPU:0'):
        chatbot_model = TransformerModel(vocab, constants.NUM_LAYERS, constants.UNITS, constants.D_MODEL,
                                         constants.NUM_HEADS, constants.DROPOUT, 'chatbot_transformer.h5')
        chatbot_model.train(train, dev)

        chatbot_model.predict('hello')


if __name__ == '__main__':
    main()
