# dataset directory for training
ds_train_dir = "dataset/train"

# dataset directory for testing
ds_test_dir = "dataset/test"

# batch size for training and testing
ds_batch_size = 32

# random seed for shuffling the data
ds_batch_seed = 36  

validate_split = 0.1

# maximum number of words to be used in the vocabulary
max_features = 10000

# output sequence length of vectorize layer
sequence_length = 400  

# dimension of the dense embedding
embedding_dim = 64

# the trainning times
epochs = 11