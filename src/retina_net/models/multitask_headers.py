import numpy as np
import tensorflow as tf

keras = tf.keras


class ClsHeader(keras.Model):

    def __init__(self, header_config):
        super(ClsHeader, self).__init__()

        self.anchors_per_location = header_config['anchors_per_location']
        self.num_classes = header_config['num_classes']

        dropout_rate = header_config['dropout_rate']

        l2_norm_rate = header_config['l2_norm_rate']

        # Classification Header (Naming convention compatible with fizyr
        # keras-retinanet)
        self.conv_1 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_classification_0',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_1 = keras.layers.Dropout(
            name='drop1', rate=dropout_rate)

        self.conv_2 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_classification_1',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_2 = keras.layers.Dropout(
            name='drop2', rate=dropout_rate)

        self.conv_3 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_classification_2',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_3 = keras.layers.Dropout(
            name='drop3', rate=dropout_rate)

        self.conv_4 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_classification_3',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_4 = keras.layers.Dropout(
            name='drop4', rate=dropout_rate)

        cls_bias_initializer = np.zeros(self.num_classes + 1)
        cls_bias_initializer[:-1] = -np.log((1.0 - 0.01) / 0.01)
        cls_bias_initializer = np.tile(
            cls_bias_initializer,
            self.anchors_per_location)

        self.cls_out = keras.layers.Conv2D(
            self.anchors_per_location * (
                self.num_classes + 1),
            (1,
             1),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer=keras.initializers.constant(cls_bias_initializer),
            name='pyramid_classification')

        self.relu = keras.layers.ReLU()

    def call(self, input_tensor, mc_dropout_enabled):
        num_input_pixels = tf.shape(input_tensor)[
            1] * tf.shape(input_tensor)[2]

        x = self.conv_1(input_tensor)
        x = self.relu(x)
        x = self.drop_1(x, training=mc_dropout_enabled)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.drop_2(x, training=mc_dropout_enabled)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.drop_3(x, training=mc_dropout_enabled)

        x = self.conv_4(x)
        x = self.relu(x)
        x = self.drop_4(x, training=mc_dropout_enabled)

        cls_out = self.cls_out(x)
        cls_out = tf.reshape(
            cls_out,
            [-1, self.anchors_per_location * num_input_pixels, self.num_classes + 1])

        return cls_out


class RegHeader(keras.Model):

    def __init__(self, header_config):
        super(RegHeader, self).__init__()

        self.anchors_per_location = header_config['anchors_per_location']

        dropout_rate = header_config['dropout_rate']

        l2_norm_rate = header_config['l2_norm_rate']

        # Regression Header
        self.conv_1 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_regression_0',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_1 = keras.layers.Dropout(
            name='drop1', rate=dropout_rate)

        self.conv_2 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_regression_1',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_2 = keras.layers.Dropout(
            name='drop2', rate=dropout_rate)

        self.conv_3 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_regression_2',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_3 = keras.layers.Dropout(
            name='drop3', rate=dropout_rate)

        self.conv_4 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_regression_3',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_4 = keras.layers.Dropout(
            name='drop4', rate=dropout_rate)

        self.reg_out = keras.layers.Conv2D(
            self.anchors_per_location * 4,
            (1,
             1),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_regression')

        self.relu = keras.layers.ReLU()

    def call(self, input_tensor, mc_dropout_enabled):
        num_input_pixels = tf.shape(input_tensor)[
            1] * tf.shape(input_tensor)[2]

        x = self.conv_1(input_tensor)
        x = self.relu(x)
        x = self.drop_1(x, training=mc_dropout_enabled)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.drop_2(x, training=mc_dropout_enabled)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.drop_3(x, training=mc_dropout_enabled)

        reg_out = self.reg_out(x)
        reg_out = tf.reshape(
            reg_out,
            [-1, self.anchors_per_location * num_input_pixels, 4])

        return reg_out


class CovHeader(keras.Model):

    def __init__(self, header_config):
        super(CovHeader, self).__init__()
        self.anchors_per_location = header_config['anchors_per_location']

        dropout_rate = header_config['dropout_rate']
        l2_norm_rate = header_config['l2_norm_rate']

        # Covariance Header
        self.conv_1 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_cov_0',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_1 = keras.layers.Dropout(
            name='drop1', rate=dropout_rate)

        self.conv_2 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_cov_1',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))
        self.drop_2 = keras.layers.Dropout(
            name='drop2', rate=dropout_rate)

        self.conv_3 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_cov_2',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_3 = keras.layers.Dropout(
            name='drop3', rate=dropout_rate)

        self.conv_4 = keras.layers.Conv2D(
            256,
            (3,
             3),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer='he_normal',
            name='pyramid_cov_3',
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate))

        self.drop_4 = keras.layers.Dropout(
            name='drop4', rate=dropout_rate)
        # Number of elements required to describe an NxN covariance matrix is
        # computed as:  (N * (N + 1)) / 2

        cov_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=1e-6)
        self.cov_out = keras.layers.Conv2D(
            self.anchors_per_location * 10,
            (1,
             1),
            strides=(
                1,
                1),
            padding='same',
            kernel_initializer=cov_init,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name='pyramid_cov')

        self.relu = keras.layers.ReLU()

    def call(self, input_tensor, mc_dropout_enabled):
        num_input_pixels = tf.shape(input_tensor)[
            1] * tf.shape(input_tensor)[2]

        x = self.conv_1(input_tensor)
        x = self.relu(x)
        x = self.drop_1(x, training=mc_dropout_enabled)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.drop_2(x, training=mc_dropout_enabled)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.drop_3(x, training=mc_dropout_enabled)

        x = self.conv_4(x)
        x = self.relu(x)
        x = self.drop_4(x, training=mc_dropout_enabled)

        cov_out = self.cov_out(x)
        cov_out = tf.reshape(
            cov_out, [-1, self.anchors_per_location * num_input_pixels, 10])

        return cov_out
