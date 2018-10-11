import tensorflow as tf


def casual_conv1d(inputs, kernel_size, filters, dilation,
                  activation, name=None):
    with tf.variable_scope(name, default_name='casual_conv1d'):
        init = tf.initializers.variance_scaling(scale=2, mode='fan_in')
        reg = tf.contrib.layers.l2_regularizer(1.0)
        inputs = tf.pad(inputs,
                        [(0, 0), ((kernel_size - 1) * dilation, 0), (0, 0)])
        layer = tf.layers.conv1d(inputs,
                                 kernel_size=kernel_size,
                                 filters=filters,
                                 dilation_rate=dilation,
                                 activation=activation,
                                 padding='valid',
                                 kernel_initializer=init,
                                 kernel_regularizer=reg)
    return layer


def res_block(inputs, kernel_size, dilation, skip_filters,
              dilation_filters, res_filters, name=None):
    with tf.variable_scope(name, default_name='res_block'):
        init = tf.initializers.variance_scaling(scale=2, mode='fan_in')
        reg = tf.contrib.layers.l2_regularizer(1.0)
        filter = casual_conv1d(inputs,
                               kernel_size,
                               dilation_filters,
                               dilation,
                               activation=tf.nn.tanh)
        gate = casual_conv1d(inputs,
                             kernel_size,
                             dilation_filters,
                             dilation,
                             activation=tf.nn.sigmoid)
        layer = filter * gate

        residual = tf.layers.conv1d(layer,
                                    filters=res_filters,
                                    kernel_size=1,
                                    kernel_initializer=init,
                                    kernel_regularizer=reg,
                                    padding='same')
        residual += inputs

        skip = tf.layers.conv1d(layer,
                                filters=skip_filters,
                                kernel_size=1,
                                kernel_initializer=init,
                                kernel_regularizer=reg,
                                padding='same')

    return skip, residual

