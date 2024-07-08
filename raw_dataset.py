import tensorflow as tf
import config


def load_raw_train_dataset():
    return tf.keras.utils.text_dataset_from_directory(
        config.ds_train_dir,
        batch_size=config.ds_batch_size, # batch size
        validation_split=config.validate_split, # 20% of data will be used for validation
        subset='training', # only training data will be used
        seed=config.ds_batch_seed) # to ensure reproducibility


def load_raw_validation_dataset():
    return tf.keras.utils.text_dataset_from_directory(
        config.ds_train_dir,
        batch_size=config.ds_batch_size,
        validation_split=config.validate_split,
        subset='validation',
        seed=config.ds_batch_seed)


def load_raw_test_dataset():
    return tf.keras.utils.text_dataset_from_directory(
        config.ds_test_dir,
        batch_size=config.ds_batch_size)
