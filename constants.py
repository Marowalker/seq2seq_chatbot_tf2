import tensorflow as tf


MAX_LENGTH = 300

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

# For Transformer
NUM_LAYERS = 2  # 6
D_MODEL = 128  # 512
NUM_HEADS = 8
UNITS = 256  # 2048
DROPOUT = 0.5

EPOCHS = 100

TRAINED_MODELS = 'trained_models/'

IS_REBUILD = 0  # 0 or 1

