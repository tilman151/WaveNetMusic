import tensorflow as tf
import os
import numpy as np
import time
from librosa.output import write_wav

import wave_net


OUTPUT_PATH = './generated'
SUMMARY_PATH = './summary'
HPARAMS = tf.contrib.training.HParams(seconds_to_generate=1,
                                      valid_freq=10,
                                      batch_size=2,
                                      num_layers=5,
                                      lr=0.001,
                                      l2_lambda=0,
                                      bin_size=256,
                                      max_dilation=512,
                                      skip_filters=512,
                                      dilation_filters=32,
                                      res_filters=32,
                                      kernel_size=2)


def generate(output_path=OUTPUT_PATH, summary_path=SUMMARY_PATH, hparams=HPARAMS):
    tf.logging.set_verbosity(tf.logging.INFO)
    assert os.path.exists(summary_path), 'Summary directory does not exist...'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    tf.logging.log(tf.logging.INFO,
                   'Build model...')
    model = wave_net.WaveNet(hparams)
    inputs = tf.placeholder(dtype=tf.int32,
                            shape=[1, model.receptive_field])
    model.build(inputs)

    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    tf.logging.log(tf.logging.INFO,
                   'Start session...')
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(summary_path)
        saver.restore(sess, ckpt)

        num_samples = hparams.seconds_to_generate * 16000
        initial_input = np.random.randint(low=0,
                                          high=hparams.bin_size,
                                          size=[1, model.receptive_field])
        samples = np.zeros([1, num_samples])
        samples[0, 0:model.receptive_field] = initial_input

        start = time.time()

        for i in range(model.receptive_field, num_samples):
            tf.logging.log_every_n(tf.logging.INFO,
                                   'Generated sample %d/%d' % (i, num_samples),
                                   n=100)
            generated = sess.run(model.generated,
                                 feed_dict={inputs: samples[:, (i - model.receptive_field):i]})
            samples[0, i] = generated[0, -1].argmax(axis=-1)

        end = time.time()

    print('Generated %d in %.1f seconds...' % (hparams.seconds_to_generate, end - start))
    audio = dequantize(samples, hparams.bin_size).squeeze()
    print('Write file...')
    write_wav(os.path.join(output_path, 'audio.wav'), audio, 16000)


def dequantize(audio, bin_size):
    bin_size = np.float32(bin_size - 1)
    audio = 2 * (audio / bin_size) - 1
    audio = np.sign(audio) * (1.0 / bin_size) * \
                             ((1.0 + bin_size) ** np.abs(audio) - 1.0)

    return audio


if __name__ == '__main__':
    generate()
