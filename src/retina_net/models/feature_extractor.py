import tensorflow as tf

keras = tf.keras


class FeatureExtractor(keras.Model):
    """
    ResNet 50 backbone feature extractor. Batch norm is frozen.
    """

    def __init__(self, feature_extractor_config):
        super(FeatureExtractor, self).__init__()

        l2_norm_rate = feature_extractor_config['l2_norm_rate']
        use_bias = feature_extractor_config['use_bias']

        self.conv_1 = keras.layers.Conv2D(
            64,
            (7,
             7),
            strides=(
                2,
                2),
            padding='valid',
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name='conv1')

        self.bn_1 = keras.layers.BatchNormalization(name='bn_conv1')
        self.pool_1_pad = keras.layers.ZeroPadding2D(
            padding=(1, 2), name='pool1_pad')
        self.pool_1 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        # Stage 2
        self.conv_block_2a = ConvBlock(3,
                                       [64,
                                        64,
                                        256],
                                       stage=2,
                                       block='a',
                                       l2_norm_rate=l2_norm_rate,
                                       strides=(1,
                                                1),
                                       use_bias=use_bias)
        self.identity_block_2b = IdentityBlock(
            3, [64, 64, 256], stage=2, block='b', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_2c = IdentityBlock(
            3, [64, 64, 256], stage=2, block='c', l2_norm_rate=l2_norm_rate, use_bias=use_bias)

        # Stage 3
        self.conv_block_3a = ConvBlock(3,
                                       [128,
                                        128,
                                        512],
                                       stage=3,
                                       block='a',
                                       l2_norm_rate=l2_norm_rate,
                                       use_bias=use_bias)
        self.identity_block_3b = IdentityBlock(
            3, [128, 128, 512], stage=3, block='b', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_3c = IdentityBlock(
            3, [128, 128, 512], stage=3, block='c', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_3d = IdentityBlock(
            3, [128, 128, 512], stage=3, block='d', l2_norm_rate=l2_norm_rate, use_bias=use_bias)

        # Stage 4
        self.conv_block_4a = ConvBlock(3,
                                       [256,
                                        256,
                                        1024],
                                       stage=4,
                                       block='a',
                                       l2_norm_rate=l2_norm_rate,
                                       use_bias=use_bias)
        self.identity_block_4b = IdentityBlock(
            3, [256, 256, 1024], stage=4, block='b', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_4c = IdentityBlock(
            3, [256, 256, 1024], stage=4, block='c', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_4d = IdentityBlock(
            3, [256, 256, 1024], stage=4, block='d', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_4e = IdentityBlock(
            3, [256, 256, 1024], stage=4, block='e', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_4f = IdentityBlock(
            3, [256, 256, 1024], stage=4, block='f', l2_norm_rate=l2_norm_rate, use_bias=use_bias)

        # Stage 5
        self.conv_block_5a = ConvBlock(3,
                                       [512,
                                        512,
                                        2048],
                                       stage=5,
                                       block='a',
                                       l2_norm_rate=l2_norm_rate,
                                       use_bias=use_bias)
        self.identity_block_5b = IdentityBlock(
            3, [512, 512, 2048], stage=5, block='b', l2_norm_rate=l2_norm_rate, use_bias=use_bias)
        self.identity_block_5c = IdentityBlock(
            3, [512, 512, 2048], stage=5, block='c', l2_norm_rate=l2_norm_rate, use_bias=use_bias)

        # Activation
        self.relu = keras.layers.ReLU()

    def call(self, input_tensor):

        # Stage 1
        x = self.conv_1(input_tensor)
        x = self.bn_1(x, training=False)
        x = self.relu(x)
        x = self.pool_1_pad(x)
        x = self.pool_1(x)

        # Stage 2
        x = self.conv_block_2a(x)
        x = self.identity_block_2b(x)
        x = self.identity_block_2c(x)

        # Stage 3
        x = self.conv_block_3a(x)
        map_3 = tf.identity(x)
        x = self.identity_block_3b(x)
        x = self.identity_block_3c(x)
        x = self.identity_block_3d(x)

        # Stage 4
        x = self.conv_block_4a(x)
        map_4 = tf.identity(x)
        x = self.identity_block_4b(x)
        x = self.identity_block_4c(x)
        x = self.identity_block_4d(x)
        x = self.identity_block_4e(x)
        x = self.identity_block_4f(x)

        # Stage 5
        x = self.conv_block_5a(x)
        x = self.identity_block_5b(x)
        x = self.identity_block_5c(x)

        return x, map_4, map_3


class IdentityBlock(keras.Model):

    def __init__(
            self,
            kernel_size,
            filters,
            stage,
            block,
            l2_norm_rate,
            use_bias=False):
        super(IdentityBlock, self).__init__()

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.is_training = keras.backend.learning_phase()

        filters1, filters2, filters3 = filters

        self.conv_1 = keras.layers.Conv2D(
            filters1,
            (1,
             1),
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2a')

        self.bn_1 = keras.layers.BatchNormalization(name=bn_name_base + '2a')

        self.conv_2 = keras.layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2b')

        self.bn_2 = keras.layers.BatchNormalization(name=bn_name_base + '2b')

        self.conv_3 = keras.layers.Conv2D(
            filters3,
            (1,
             1),
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2c')
        self.bn_3 = keras.layers.BatchNormalization(name=bn_name_base + '2c')

        self.relu = keras.layers.ReLU()

    def call(self, input_tensor):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
        """
        x = self.conv_1(input_tensor)
        x = self.bn_1(x, training=False)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=False)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x, training=False)
        x = keras.layers.add([x, input_tensor])
        x = self.relu(x)
        return x


class ConvBlock(keras.Model):

    def __init__(
            self,
            kernel_size,
            filters,
            stage,
            block,
            l2_norm_rate,
            strides=(
                2,
                2),
            use_bias=False):
        super(ConvBlock, self).__init__()

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.is_training = keras.backend.learning_phase()
        filters1, filters2, filters3 = filters

        self.conv_1 = keras.layers.Conv2D(
            filters1,
            (1,
             1),
            strides=strides,
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2a')
        self.bn_1 = keras.layers.BatchNormalization(name=bn_name_base + '2a')

        self.conv_2 = keras.layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2b')
        self.bn_2 = keras.layers.BatchNormalization(name=bn_name_base + '2b')

        self.conv_3 = keras.layers.Conv2D(
            filters3,
            (1,
             1),
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '2c')
        self.bn_3 = keras.layers.BatchNormalization(name=bn_name_base + '2c')

        self.shortcut = keras.layers.Conv2D(
            filters3,
            (1,
             1),
            strides=strides,
            kernel_initializer='he_normal',
            use_bias=use_bias,
            kernel_regularizer=keras.regularizers.l2(l2_norm_rate),
            name=conv_name_base + '1')

        self.bn_shortcut = keras.layers.BatchNormalization(
            name=bn_name_base + '1')

        self.relu = keras.layers.ReLU()

    def call(self, input_tensor):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the second conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        x = self.conv_1(input_tensor)
        x = self.bn_1(x, training=False)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=False)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x, training=False)
        x_shortcut = self.shortcut(input_tensor)
        x_shortcut = self.bn_shortcut(x_shortcut, training=False)
        x = keras.layers.add([x, x_shortcut])
        x = self.relu(x)
        return x
