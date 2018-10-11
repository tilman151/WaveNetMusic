import tensorflow as tf
import numpy as np

import layers as ly


class WaveNet:
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.num_layers = hparams.num_layers
        self.max_dilation = hparams.max_dilation
        self.skip_filters = hparams.skip_filters
        self.res_filters = hparams.res_filters
        self.dilation_filters = hparams.dilation_filters
        self.kernel_size = hparams.kernel_size
        self.bin_size = hparams.bin_size
        self.lr = hparams.lr
        self.l2_lambda = hparams.l2_lambda

        dilation_exp = int(np.log2(self.max_dilation)) + 1
        dilations = [2 ** (i % dilation_exp) for i in range(self.num_layers)]
        self.receptive_field = (self.kernel_size - 1) * sum(dilations) \
                                + self.kernel_size

        self.generated = None
        self.dilations = None
        self.output = None
        self.loss = None
        self.train_op = None
        self.summary = None
        self.global_step = tf.train.get_or_create_global_step()

    def build(self, inputs):
        inputs = tf.one_hot(inputs, depth=self.bin_size)
        self.generated = self._network(inputs)
        self.output = tf.nn.softmax(self.generated, axis=-1)
        self.loss = self._loss(inputs, self.generated)
        self.train_op = self._train_op(self.loss)

        self.summary = tf.summary.merge_all()

    def build_generator(self):
        pass

    def _network(self, inputs):
        init = tf.initializers.variance_scaling(scale=2, mode='fan_in')
        reg = tf.contrib.layers.l2_regularizer(1.0)

        with tf.variable_scope('preprocessing'):
            layers = [(0, ly.casual_conv1d(inputs,
                                           kernel_size=self.kernel_size,
                                           filters=self.res_filters,
                                           dilation=1,
                                           activation=None))]

        with tf.variable_scope('residual_stack'):
            max_dilation_exp = int(np.log2(self.max_dilation)) + 1
            for i in range(self.num_layers):
                dilation = 2 ** (i % max_dilation_exp)
                layers.append(ly.res_block(
                                       layers[-1][1],
                                       kernel_size=self.kernel_size,
                                       dilation=dilation,
                                       skip_filters=self.skip_filters,
                                       dilation_filters=self.dilation_filters,
                                       res_filters=self.res_filters))

        with tf.variable_scope('postprocessing'):
            skips = [l[0] for l in layers[1:]]
            skips = tf.add_n(skips)
            skips = tf.nn.relu(skips)
            skips = tf.layers.conv1d(skips,
                                     filters=self.skip_filters,
                                     kernel_size=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=init,
                                     kernel_regularizer=reg)
            skips = tf.layers.conv1d(skips,
                                     filters=256,
                                     kernel_size=1,
                                     padding='same',
                                     kernel_initializer=init,
                                     kernel_regularizer=reg)

        return skips

    def _loss(self, real, generated):
        with tf.name_scope('loss'):
            labels = real[:, (self.receptive_field + 1):]
            predictions = generated[:, self.receptive_field:-1]
            tf.summary.histogram('generated', tf.argmax(predictions, axis=-1))
            tf.summary.histogram('real', tf.argmax(labels, axis=-1))

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        labels=labels,
                                                        logits=predictions)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('cross_entropy', loss)

            if self.l2_lambda > 0:
                l2_loss = tf.add_n(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                tf.summary.scalar('l2', l2_loss)

                loss += self.l2_lambda * l2_loss

        return loss

    def _train_op(self, loss):
        optim = tf.train.AdamOptimizer(learning_rate=self.lr,
                                       beta1=0.5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optim.minimize(loss,
                                      global_step=self.global_step)

        return train_op
