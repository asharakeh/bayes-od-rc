import tensorflow as tf

keras = tf.keras


class SoftmaxFocalLoss(keras.losses.Loss):
    """
    Focal loss implementation in keras.
    """

    def __init__(
            self,
            gamma=2,
            alpha=0.5,
            temperature=1.0,
            label_smoothing_epsilon=0.001,
            reduction=keras.losses.Reduction.NONE):
        super(SoftmaxFocalLoss, self).__init__(reduction=reduction,
                                               name='focal_loss')

        self.cross_entropy = keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing_epsilon,
            reduction=self.reduction)

        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature

    def call(self, target_tensor, prediction_tensor):
        # Divide logits by temperature.
        # A temperature value > 1.0 defuses predictions, making the network less confident.
        # A temperature value < 1.0 but > 0.0 peaks predictions, making the
        # network more confident.
        prediction_tensor = tf.divide(prediction_tensor,
                                      self.temperature,
                                      name='scale_logit')

        # Compute background and positive targets to get weighting_factor.
        # This is not the exact formulation in the paper, but a workable
        # approximation.

        stable_prediction = prediction_tensor - \
            tf.reduce_max(prediction_tensor, axis=2, keepdims=True)
        stable_softmax = tf.nn.softmax(stable_prediction)
        stable_softmax = tf.reduce_sum(stable_softmax * target_tensor, axis=2)

        per_row_cross_ent = self.cross_entropy(target_tensor,
                                               prediction_tensor)
        # Gamma modulating factor
        focusing_factor = tf.pow(
            (1 - stable_softmax), self.gamma)

        # Alpha factor, shared among all classes
        negative_mask = target_tensor[:, :, -1]
        positive_mask = 1 - negative_mask

        alpha_modulating_factor = self.alpha * \
            positive_mask + (1 - self.alpha) * negative_mask

        return alpha_modulating_factor * focusing_factor * per_row_cross_ent
