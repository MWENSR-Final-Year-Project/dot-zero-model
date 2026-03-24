import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.bn2(self.conv2(x), training=training)
        x += residual
        return tf.nn.relu(x)


class DotZeroNet(tf.keras.Model):
    def __init__(
        self, board_size: int = 5, channels: int = 64, num_res_blocks: int = 6
    ):
        super().__init__()
        self.n = board_size
        self.spatial = board_size + 1
        self.action_size = 2 * board_size * (board_size + 1)

        self.conv = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

        self.res_blocks = [ResidualBlock(channels) for _ in range(num_res_blocks)]

        self.policy_conv = tf.keras.layers.Conv2D(2, 1)
        self.policy_bn = tf.keras.layers.BatchNormalization()

        self.value_conv = tf.keras.layers.Conv2D(1, 1)
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.value_fc1 = tf.keras.layers.Dense(64)
        self.value_fc2 = tf.keras.layers.Dense(1)

    def call(self, x, legal_mask=None, training=False):
        x = tf.nn.relu(self.bn(self.conv(x), training=training))
        for block in self.res_blocks:
            x = block(x, training=training)

        p = self.policy_bn(self.policy_conv(x), training=training)
        batch_size = tf.shape(x)[0]
        horizontal_logits = tf.reshape(p[:, :, : self.n, 0], [batch_size, -1])
        vertical_logits = tf.reshape(p[:, : self.n, :, 1], [batch_size, -1])
        p_flat = tf.concat([horizontal_logits, vertical_logits], axis=1)

        if legal_mask is not None:
            large_neg = tf.fill(tf.shape(p_flat), float("-inf"))
            p_flat = tf.where(legal_mask == 0, large_neg, p_flat)

        v = tf.nn.relu(self.value_bn(self.value_conv(x), training=training))
        v = tf.reshape(v, [batch_size, -1])
        v = tf.nn.relu(self.value_fc1(v))
        v = tf.nn.tanh(self.value_fc2(v))

        return p_flat, tf.squeeze(v, axis=-1)
