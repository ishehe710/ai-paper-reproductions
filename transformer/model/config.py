EMBED_SIZE = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_LAYERS = 2
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
D_MODEL = 128 # for embedding layer
MAX_LENGTH_ENCODING = 512 # for positional encoding
DATASET_FILENAME = "./transformer/data/eng_-french.csv"

# extracting dataest columns
ENGLISH = 'English words/sentences'
FRENCH = 'French words/sentences'

# training and validation
TEST_PERCENTAGE = 0.2
VAL_PERCENTAGE = 0.2
