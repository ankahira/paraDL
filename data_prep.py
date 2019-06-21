import tensorflow as tf
import numpy as np
import os


def get_data_files(data_dir):
    n_train_files = None
    samples_per_file = 16
    input_shape = (128, 128, 128, 1)
    target_size = 3

    batch_size = 8
    n_epochs = 4
    train_data_dir = data_dir  # os.path.join(data_dir, 'train')
    train_files = [os.path.join(train_data_dir, f)
                   for f in os.listdir(train_data_dir)
                   if f.endswith('tfrecords')][:n_train_files]

    n_train_files = len(train_files)
    n_train = n_train_files * samples_per_file
    print('Loaded %i training samples in %i TFRecord files' % (n_train, n_train_files))

    train_dataset = (tf.data.TFRecordDataset(filenames=train_files).map(_parse_data).batch(batch_size))


def _parse_data(sample_proto):
    parsed_example = tf.parse_single_example(
        sample_proto,
        features={'3Dmap': tf.FixedLenFeature([], tf.string),
                    'unitPar': tf.FixedLenFeature([], tf.string),
                    'physPar': tf.FixedLenFeature([], tf.string)}
    )
    # Decode the data and normalize
    data = tf.decode_raw(parsed_example['3Dmap'], tf.float32)
    data = tf.reshape(data, [128, 128, 128, 1])
    data /= (tf.reduce_sum(data) / 128**3)
    # Decode the targets
    label = tf.decode_raw(parsed_example['unitPar'], tf.float32)
    return data, label


def construct_dataset(filenames, batch_size, n_epochs, shuffle_buffer_size=128):
    return (tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=4)
            .shuffle(len(filenames), reshuffle_each_iteration=True)
            .repeat(n_epochs)
            .map(_parse_data)
            .shuffle(shuffle_buffer_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(4))


if __name__ == "__main__":
    data_dir = "data"

    get_data_files(data_dir)

    # tf.enable_eager_execution()
    # dataset = tf.data.TFRecordDataset(file_path)
    # iterator = dataset.make_one_shot_iterator()
    # count = 0
    # for record in iterator:
    #     parsed_example = tf.parse_single_example(
    #         record,
    #         features={'3Dmap': tf.FixedLenFeature([], tf.string),
    #                   'unitPar': tf.FixedLenFeature([], tf.string),
    #                   'physPar': tf.FixedLenFeature([], tf.string)}
    #     )
    #     # Decode the data and normalize
    #     data = tf.decode_raw(parsed_example['3Dmap'], tf.float32)
    #     data = tf.reshape(data, [128, 128, 128, 4])
    #     data /= (tf.reduce_sum(data) / 128 ** 3)
    #     # Decode the targets
    #     label = tf.decode_raw(parsed_example['unitPar'], tf.float32)
    #     print(label)
    #
    #     count += 1
    #
    # print(count)





