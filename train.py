import tensorflow as tf
import os
import shutil
import time

import audio_inputs
import wave_net


TRAIN_PATH = './data/train'
VALID_PATH = './data/valid'
SUMMARY_PATH = './summary'
HPARAMS = tf.contrib.training.HParams(max_epochs=10000,
                                      valid_freq=1,
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


def train(train_path=TRAIN_PATH, valid_path=VALID_PATH,
          summary_path=SUMMARY_PATH, hparams=HPARAMS):
    if os.path.exists(summary_path):
        shutil.rmtree(summary_path)
    os.mkdir(summary_path)

    tf.logging.log(tf.logging.INFO,
                   'Build inputs...')
    with tf.device('/cpu:0'):
        inputs = audio_inputs.AudioInputs(train_path,
                                          valid_path,
                                          hparams.batch_size,
                                          repeat=hparams.valid_freq)
        audio, train_init, valid_init = inputs.get_next()

    tf.logging.log(tf.logging.INFO,
                   'Build model...')

    model = wave_net.WaveNet(hparams)
    model.build(audio)

    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    tf.logging.log(tf.logging.INFO,
                   'Start session...')
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_path,
                                               tf.get_default_graph())

        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer(),
                  train_init])

        for i in range(hparams.max_epochs):
            epoch_step = 0
            while True:
                try:
                    step, summary, loss, _ = sess.run([model.global_step,
                                                       model.summary,
                                                       model.loss,
                                                       model.train_op])
                except tf.errors.OutOfRangeError:
                    break

                tf.logging.log_every_n(
                       tf.logging.INFO,
                       'Train step %d:%d, loss %.5f' % (i, epoch_step, loss),
                       n=100)
                summary_writer.add_summary(summary, global_step=step)
                epoch_step += 1

            saver.save(sess,
                       save_path=os.path.join(summary_path, 'model_ckpt'),
                       global_step=i)

            if i % hparams.valid_freq == 0:
                tf.logging.log(tf.logging.INFO,
                               'Validating...')
                sess.run([valid_init])
                log_prob = 0
                while True:
                    try:
                        log_prob += sess.run(model.loss)
                    except tf.errors.OutOfRangeError:
                        break

                tf.logging.log(tf.logging.INFO,
                               'Validation loss: %.4f' % log_prob)
                valid_summary = tf.summary.scalar('valid_log_prob', log_prob)
                summary_writer.add_summary(sess.run(valid_summary))

                sess.run(train_init)


if __name__ == '__main__':
    train(summary_path='./summary/%d' % int(time.time()))
