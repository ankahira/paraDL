import tensorflow as tf
import numpy as np


def read_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    single_example = raw_dataset.make_one_shot_iterator().get_next()
    parsed_example = tf.parse_single_example(
        single_example,
        features={
            "data_raw": tf.FixedLenFeature([], tf.string),
            "label_raw": tf.FixedLenFeature([], tf.string)
        }
    )

    NbodySimuDecode = tf.decode_raw(parsed_example['data_raw'], tf.float64)
    labelDecode = tf.decode_raw(parsed_example['label_raw'], tf.float64)
    NbodySimus = tf.reshape(NbodySimuDecode, [64, 64, 64])
    NbodySimus /= (tf.reduce_sum(NbodySimus) / 64 ** 3 + 0.)
    NbodySimuAddDim = tf.expand_dims(NbodySimus, 3)
    label = tf.reshape(labelDecode, [2])
    labelAddDim = (label - tf.constant([2.995679839999998983e-01, 8.610806619999996636e-01],
                                       dtype=tf.float64)) / tf.constant(
        [2.905168635566176411e-02, 4.023372385668218254e-02], dtype=tf.float64)
    print(NbodySimuAddDim.shape)

    return NbodySimuAddDim, labelAddDim


if __name__ == "__main__":
    tf.enable_eager_execution()

    tf.enable_eager_execution()
    path = "data/cosmoUniverse_2019_02_4parE-dim128_cube_nT4-rec1000.tfrecords"

    data, label  = read_tfrecord(path)


