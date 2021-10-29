import tensorflow as tf

MAX_LENGTH = 50

DATA = 'data/'

DATA_TRAIN = DATA + 'train/dialogues_train.txt'
DATA_DEV = DATA + 'validation/dialogues_validation.txt'
DATA_TEST = DATA + 'test/dialogues_test.txt'
DATA_FULL = DATA + 'ijcnlp_dailydialog/dialogues_text.txt'

PICKLE = DATA + 'pickle/'

VOCAB = PICKLE + 'vocab.pkl'

START = '$START$'
END = '$END$'
UNK = '$UNK$'

strategy = tf.distribute.get_strategy()
# For tf.data.Dataset
BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
BUFFER_SIZE = 20000

topic = {
    1: 'i am living an ordinary fife .',
    2: 'i am at school .',
    3: 'i like culture and education .',
    4: 'i have attitude and emotion .',
    5: 'i am in a relationship .',
    6: 'i like tourism .',
    7: 'i like health advices .',
    8: 'i am at work .',
    9: 'i like politics .',
    10: 'i like finance .'
}

act = {
    1: 'i am informative .',
    2: 'i am asking a question .',
    3: 'i am directive .',
    4: 'i am commissive .'
}

emotion = {
    0: 'i have no emotion .',
    1: 'i am angry .',
    2: 'i am disgusted .',
    3: 'i am afraid .',
    4: 'i am happy .',
    5: 'i am sad .',
    6: 'i am surprised .'
}

# For Transformer
NUM_LAYERS = 2  # 6
D_MODEL = 128  # 512
NUM_HEADS = 8
UNITS = 256  # 2048
DROPOUT = 0.5

EPOCHS = 100

TRAINED_MODELS = 'trained_models/'

SEQ2SEQ = TRAINED_MODELS + 'seq2seq/'

IS_REBUILD = 1  # 0 or 1
