import tensorflow as tf
import numpy as np

file_path = "data/cosmoUniverse_2019_02_4parE-dim128_cube_nT4-rec1000.tfrecords"


if __name__ == "__main__":
    tf.enable_eager_execution()
    dataset = tf.data.TFRecordDataset(file_path)
    iterator = dataset.make_one_shot_iterator()
    count = 0
    for record in iterator:
        parsed_example = tf.parse_single_example(
            record,
            features={'3Dmap': tf.FixedLenFeature([], tf.string),
                      'unitPar': tf.FixedLenFeature([], tf.string),
                      'physPar': tf.FixedLenFeature([], tf.string)}
        )
        # Decode the data and normalize
        data = tf.decode_raw(parsed_example['3Dmap'], tf.float32)
        data = tf.reshape(data, [128, 128, 128, 4])
        data /= (tf.reduce_sum(data) / 128 ** 3)
        # Decode the targets
        label = tf.decode_raw(parsed_example['unitPar'], tf.float32)
        print(data.shape)
        print(label.shape)
        break




