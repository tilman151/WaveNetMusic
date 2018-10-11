import tensorflow as tf
import random
import os


class AudioInputs:
    def __init__(self, train_path, valid_path, batch_size,
                 bin_size=256, shuffle=True, repeat=True):
        self.batch_size = batch_size
        self.bin_size = bin_size

        self.train_set = self._build_dataset(train_path,
                                             batch_size,
                                             repeat,
                                             shuffle)
        self.valid_set = self._build_dataset(valid_path,
                                             batch_size,
                                             repeat=False,
                                             shuffle=False)

    def _build_dataset(self, input_path, batch_size, repeat, shuffle):
        files = [os.path.join(input_path, p) for p in os.listdir(input_path)]
        files = [p for p in files if os.path.isfile(p)]
        random.shuffle(files)
        dataset = tf.data.TFRecordDataset(files,
                                          num_parallel_reads=4)
        dataset = dataset.map(self._parse_sample,
                              num_parallel_calls=4)
        dataset = dataset.map(self._quantize,
                              num_parallel_calls=4)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=4000)
        if repeat:
            dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)

        return dataset

    def get_next(self, name='batch'):
        iterator = tf.data.Iterator.from_structure(self.train_set.output_types,
                                                   self.train_set.output_shapes)
        audio = iterator.get_next(name)
        audio.set_shape([self.batch_size, 10000])

        with tf.name_scope('training_set'):
            train_init = iterator.make_initializer(self.train_set)
        with tf.name_scope('validation_set'):
            valid_init = iterator.make_initializer(self.valid_set)

        return audio, train_init, valid_init

    @staticmethod
    def _parse_sample(serialized_sample):
        context = {'seq_len': tf.FixedLenFeature([], tf.int64)}
        sequence = {'wave': tf.FixedLenSequenceFeature([], tf.float32)}
        context, sequence = tf.parse_single_sequence_example(
            serialized_sample,
            context_features=context,
            sequence_features=sequence)

        return sequence['wave']

    def _quantize(self, audio):
        bin_size = tf.cast(self.bin_size - 1, tf.float32)
        quant = tf.sign(audio) * (tf.log1p(bin_size * tf.abs(audio))) / \
                                 (tf.log1p(bin_size))
        quant = tf.cast((quant + 1) / 2 * bin_size + 0.5, tf.int32)

        return quant

    def dequantize(self, audio):
        bin_size = tf.cast(self.bin_size - 1, tf.float32)
        audio = 2 * (tf.to_float(audio) / bin_size) - 1
        audio = tf.sign(audio) * (1.0 / bin_size) * \
                                 ((1.0 + bin_size) ** tf.abs(audio) - 1.0)

        return audio
