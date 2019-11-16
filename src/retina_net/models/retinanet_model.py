import tensorflow as tf
import tensorflow_probability as tfp

import src.core.constants as constants

from src.core.losses import SoftmaxFocalLoss
from src.retina_net.models.feature_extractor import FeatureExtractor
from src.retina_net.models.feature_decoder import FeatureDecoder
from src.retina_net.models.multitask_headers import ClsHeader
from src.retina_net.models.multitask_headers import RegHeader
from src.retina_net.models.multitask_headers import CovHeader
from src.retina_net.anchor_generator import box_utils

keras = tf.keras


class RetinaNetModel(keras.Model):

    def __init__(self, model_config):
        super(RetinaNetModel, self).__init__()

        self.prediction_dict = None

        # Create loss instants
        loss_config = model_config['losses']
        self.focal_loss = SoftmaxFocalLoss(
            gamma=2.0,
            label_smoothing_epsilon=loss_config['label_smoothing_epsilon'],
            reduction=keras.losses.Reduction.NONE)

        self.huber_loss = keras.losses.Huber(
            reduction=keras.losses.Reduction.NONE, name='huber_loss')

        self.loss_names = loss_config['loss_names']
        self.loss_weights = loss_config['loss_weights']

        # Feature Extractor
        with tf.name_scope(model_config['feature_extractor']['name']):
            self.feature_extractor = FeatureExtractor(
                model_config['feature_extractor'])

        # Feature Decoder
        with tf.name_scope(model_config['feature_extractor']['name']):
            self.feature_decoder = FeatureDecoder(
                model_config['feature_decoder'])

        # Headers
        self.compute_cls = 'classification' in model_config['output_names']
        self.compute_reg = 'regression' in model_config['output_names']
        self.compute_covar = 'regression_covar' in model_config['output_names']

        if self.compute_cls:
            with tf.name_scope('classification_header'):
                self.cls_header = ClsHeader(
                    model_config['header'])
        if self.compute_reg:
            with tf.name_scope('regression_header'):
                self.reg_header = RegHeader(
                    model_config['header'])
        if self.compute_covar:
            with tf.name_scope('covariance_header'):
                self.cov_header = CovHeader(
                    model_config['header'])

        self.mc_dropout_samples = model_config['mc_dropout_samples']

    def call(self, input_tensor, train_val_test='training'):

        self.prediction_dict = dict()
        map_5, map_4, map_3 = self.feature_extractor(input_tensor)
        decoder_pyramid_layers = self.feature_decoder(map_5, map_4, map_3)

        if train_val_test == 'testing':
            if self.mc_dropout_samples > 1.0:
                mc_dropout_enabled = True
            else:
                mc_dropout_enabled = False
            decoder_pyramid_layers = [
                tf.tile(
                    layer, [
                        self.mc_dropout_samples, 1, 1, 1])for layer in decoder_pyramid_layers]
            if self.compute_cls:
                cls_out_list = [
                    self.cls_header(
                        pyramid_layer,
                        mc_dropout_enabled) for pyramid_layer in
                    decoder_pyramid_layers]

                cls_out = tf.concat(cls_out_list, axis=1)
                self.prediction_dict.update(
                    {constants.ANCHORS_CLASS_PREDICTIONS_KEY: cls_out})

            if self.compute_reg:
                reg_out_list = [
                    self.reg_header(
                        pyramid_layer,
                        mc_dropout_enabled) for pyramid_layer in
                    decoder_pyramid_layers]
                reg_out = tf.concat(reg_out_list, axis=1)
                self.prediction_dict.update(
                    {constants.ANCHORS_BOX_PREDICTIONS_KEY: reg_out})

            if self.compute_covar:
                covar_out_list = [
                    self.cov_header(
                        pyramid_layer,
                        mc_dropout_enabled) for pyramid_layer in
                    decoder_pyramid_layers]
                covar_out = tf.concat(covar_out_list, axis=1)
                covar_out = tfp.distributions.fill_triangular(covar_out)
                self.prediction_dict.update(
                    {constants.ANCHORS_COVAR_PREDICTIONS_KEY: covar_out})
        else:
            if train_val_test == 'training':
                mc_dropout_enabled = True
            else:
                mc_dropout_enabled = False

            if self.compute_cls:
                cls_out_list = [
                    self.cls_header(
                        pyramid_layer,
                        mc_dropout_enabled) for pyramid_layer in decoder_pyramid_layers]

                cls_out = tf.concat(cls_out_list, axis=1)
                self.prediction_dict.update(
                    {constants.ANCHORS_CLASS_PREDICTIONS_KEY: cls_out})

            if self.compute_reg:
                reg_out_list = [
                    self.reg_header(
                        pyramid_layer,
                        mc_dropout_enabled) for pyramid_layer in decoder_pyramid_layers]
                reg_out = tf.concat(reg_out_list, axis=1)
                self.prediction_dict.update(
                    {constants.ANCHORS_BOX_PREDICTIONS_KEY: reg_out})

            if self.compute_covar:
                covar_out_list = [
                    self.cov_header(
                        pyramid_layer,
                        mc_dropout_enabled)for pyramid_layer in decoder_pyramid_layers]
                covar_out = tf.concat(covar_out_list, axis=1)
                covar_out = tfp.distributions.fill_triangular(covar_out)

                self.prediction_dict.update(
                    {constants.ANCHORS_COVAR_PREDICTIONS_KEY: covar_out})

        return self.prediction_dict

    def get_loss(self,
                 sample_dict,
                 prediction_dict):
        with tf.name_scope("loss_computation"):
            # Initialize total loss to 0.0
            total_loss = tf.constant(0.0)
            loss_dict = dict()

            # Get ground truth tensors
            anchors = sample_dict[constants.ANCHORS_KEY]
            anchor_positive_mask = sample_dict[constants.POSITIVE_ANCHORS_MASK_KEY]
            anchor_positive_mask = tf.cast(
                anchor_positive_mask,
                tf.float32)
            num_positives = tf.reduce_sum(anchor_positive_mask)

            anchor_negative_mask = sample_dict[constants.NEGATIVE_ANCHOR_MASK_KEY]
            anchor_negative_mask = tf.cast(
                anchor_negative_mask,
                tf.float32)

            classification_loss_mask = anchor_positive_mask + anchor_negative_mask

            target_anchor_classes = sample_dict[constants.ANCHORS_CLASS_TARGETS_KEY]
            target_anchor_boxes = sample_dict[constants.ANCHORS_BOX_TARGETS_KEY]

            # Get prediction tensors
            predicted_anchor_classes = prediction_dict[constants.ANCHORS_CLASS_PREDICTIONS_KEY]
            predicted_anchor_boxes = prediction_dict[constants.ANCHORS_BOX_PREDICTIONS_KEY]

            with tf.name_scope('total_loss'):
                for loss in self.loss_names:
                    if loss == 'classification':
                        # Get loss weights
                        loss_weight = self.loss_weights[self.loss_names.index(
                            loss)]

                        # Compute Loss
                        anchorwise_cls_loss = self.focal_loss(
                            target_anchor_classes, predicted_anchor_classes)

                        # Normalize Loss By Number Of Anchors and multiply by loss
                        # weight
                        normalized_cls_loss = tf.reduce_sum(
                            anchorwise_cls_loss * classification_loss_mask) / tf.maximum(
                            num_positives, 1) * loss_weight

                        # Update dictionary for summaries
                        loss_dict.update(
                            {constants.CLS_LOSS_KEY: normalized_cls_loss})

                        # Update total loss
                        total_loss += normalized_cls_loss

                    elif loss == 'regression':
                        loss_weight = self.loss_weights[self.loss_names.index(
                            loss)]

                        # Compute Loss
                        element_wise_reg_loss = self.huber_loss(
                            target_anchor_boxes, predicted_anchor_boxes)

                        # Normalize Loss By Number Of Anchors and multiply by loss
                        # weight
                        normalized_reg_loss = tf.reduce_sum(
                            tf.reduce_mean(
                                element_wise_reg_loss,
                                axis=2) * anchor_positive_mask) / tf.maximum(
                            num_positives,
                            1) * loss_weight

                        # Update dictionary for summaries
                        loss_dict.update(
                            {constants.REG_LOSS_KEY: normalized_reg_loss})

                        total_loss += normalized_reg_loss

                    elif loss == 'regression_var':
                        loss_weight = self.loss_weights[self.loss_names.index(
                            loss)]

                        # Get predictions
                        predicted_boxes = box_utils.box_from_anchor_and_target_bnms(
                            anchors, predicted_anchor_boxes)

                        # Get Ground truth
                        target_boxes = box_utils.box_from_anchor_and_target_bnms(
                            anchors, target_anchor_boxes)

                        # Get estimated inverse of the cholskey decomposition
                        anchorwise_covar_predictions = prediction_dict[
                            constants.ANCHORS_COVAR_PREDICTIONS_KEY]
                        log_D = tf.linalg.diag_part(
                            anchorwise_covar_predictions)

                        # Compute Loss
                        element_wise_reg_loss = self.huber_loss(
                            target_boxes, predicted_boxes)

                        covar_compute_loss = tf.reduce_sum(
                            tf.exp(-log_D) * element_wise_reg_loss, axis=2)
                        covar_reg_loss = 0.5 * tf.reduce_sum(log_D, axis=2)

                        covar_final_loss = covar_compute_loss + covar_reg_loss

                        # Normalize Loss By Number Of Anchors and multiply by loss
                        # weight
                        normalized_reg_loss = loss_weight * \
                            tf.reduce_sum(covar_final_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)

                        # Update dictionary for summaries
                        regression_loss = tf.reduce_sum(
                            covar_compute_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)
                        regularization_loss = tf.reduce_sum(
                            covar_reg_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)

                        loss_dict.update(
                            {constants.REG_LOSS_KEY: regression_loss})
                        loss_dict.update(
                            {constants.COV_LOSS_KEY: regularization_loss})

                        total_loss = total_loss + normalized_reg_loss

                    elif loss == 'regression_covar':
                        loss_weight = self.loss_weights[self.loss_names.index(
                            loss)]

                        # Get predictions
                        predicted_boxes = box_utils.box_from_anchor_and_target_bnms(
                            anchors, predicted_anchor_boxes)

                        # Get Ground truth
                        target_boxes = box_utils.box_from_anchor_and_target_bnms(
                            anchors, target_anchor_boxes)

                        # Get estimated inverse of the cholskey decomposition
                        anchorwise_covar_predictions = prediction_dict[
                            constants.ANCHORS_COVAR_PREDICTIONS_KEY]
                        log_D = tf.linalg.diag_part(
                            anchorwise_covar_predictions)

                        L_inv = tf.linalg.set_diag(
                            anchorwise_covar_predictions, tf.ones_like(log_D))

                        fro_norm = tf.linalg.norm(L_inv, axis=(-2, -1))

                        # Compute Loss
                        element_wise_reg_loss = self.huber_loss(
                            target_boxes, predicted_boxes)

                        covar_compute_loss = fro_norm * tf.reduce_sum(
                            tf.exp(-log_D) * element_wise_reg_loss, axis=2)
                        covar_reg_loss = 0.5 * tf.reduce_sum(log_D, axis=2)

                        covar_final_loss = covar_compute_loss + covar_reg_loss

                        # Normalize Loss By Number Of Anchors and multiply by loss
                        # weight
                        normalized_reg_loss = loss_weight * \
                            tf.reduce_sum(covar_final_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)

                        # Update dictionary for summaries
                        regression_loss = tf.reduce_sum(
                            covar_compute_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)
                        regularization_loss = tf.reduce_sum(
                            covar_reg_loss * anchor_positive_mask) / tf.maximum(1.0, num_positives)

                        loss_dict.update(
                            {constants.REG_LOSS_KEY: regression_loss})
                        loss_dict.update(
                            {constants.COV_LOSS_KEY: regularization_loss})

                        total_loss = total_loss + normalized_reg_loss
                    else:
                        raise ValueError(
                            'Invalid Loss! Not implemented yet.', loss)

            return total_loss, loss_dict
