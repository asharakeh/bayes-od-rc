import numpy as np
import tensorflow as tf

from src.core.abstract_classes.anchor_generator import AnchorGenerator


class FpnAnchorGenerator(AnchorGenerator):
    """
    Class that generates anchors for FPN, based on the scale of every layer.
    """

    def __init__(self, generator_config):
        # Parse Generator Config
        super(FpnAnchorGenerator, self).__init__(generator_config)

        self.aspect_ratios = self.config['aspect_ratios']
        self.scales = self.config['scales']
        self.anchors_per_location = tf.constant(
            np.size(self.aspect_ratios, axis=0) * np.size(self.scales))

    def generate_anchors(self, im_shape, layer_number):
        aspect_ratios = self.aspect_ratios
        scales = self.scales

        im_shape = tf.cast(im_shape, tf.float32)

        stride = tf.pow(2.0, layer_number)
        u_pos = (tf.range(0, im_shape[1] / stride) + 0.5) * stride
        v_pos = (tf.range(0, im_shape[0] / stride) + 0.5) * stride
        u, v = tf.meshgrid(u_pos, v_pos)
        u = tf.reshape(u, [-1])
        v = tf.reshape(v, [-1])
        spatial_locations = tf.stack((v, u), axis=1)

        anchor_side_shape = tf.pow(2.0, layer_number + 2.0)
        anchor_dims = []
        for aspect_ratio in aspect_ratios:
            for scale in scales:
                if aspect_ratio[0] == 1 and aspect_ratio[1] == 1:
                    anchor_dims.append(
                        aspect_ratio * anchor_side_shape * scale)
                else:
                    dim_solution = tf.sqrt(
                        tf.pow(
                            anchor_side_shape,
                            2.0) /
                        np.prod(aspect_ratio))
                    anchor_dims.append(aspect_ratio * dim_solution * scale)

        num_of_locations = tf.cast(tf.shape(spatial_locations)[
            0], tf.int32)

        anchor_locations = tf_repeat(spatial_locations,
                                     [self.anchors_per_location, 1])
        anchor_dims = tf.tile(tf.stack(anchor_dims), [num_of_locations, 1])

        anchor_grid = tf.concat((anchor_locations, anchor_dims), axis=1)

        return anchor_grid

    @staticmethod
    def positive_negative_batching(
            ious,
            min_positive_iou=0.5,
            max_negative_iou=0.4):
        """
        Generates positive and negative anchors using min and max iou thresholds.
        :param ious: Batch x N x M matrix containing the iou of every anchor with every groundtruth target
        :param min_positive_iou: minimum threshold for an anchor to be positive
        :param max_negative_iou: maximum threshold for an anchor to be negative
        :return:
        """
        positive_mask = tf.greater_equal(ious, min_positive_iou)
        positive_mask = tf.reduce_any(positive_mask, axis=1)
        negative_mask = tf.less_equal(ious, max_negative_iou)
        negative_mask = tf.reduce_all(negative_mask, axis=1)

        max_iou = tf.argmax(ious, axis=1, name=None)
        return positive_mask, negative_mask, max_iou

    @staticmethod
    def generate_anchor_targets(
            anchors,
            gt_boxes,
            gt_classes,
            max_ious,
            positive_anchor_mask):
        """
        Generates anchor targets to be fed into the loss function computation.

        :param anchors: N x 4 tensor of anchors
        :param gt_boxes: M x 4 tensor of GT boxes
        :param gt_classes: M x K tensor of classes in one-hot format
        :param max_ious: N x 1 tensor of indexes of the gt which the anchor has max iou
        :param positive_anchor_mask: N x 1 tensor signifying which anchor is positive.

        :return: anchor_box_targets: N x 4 anchor regression targets
        :return: anchor_class_targets: N x k anchor classification targets
        """

        anchor_box_gt = tf.gather(gt_boxes, max_ious, axis=0)

        anchor_box_pos_v_targets = (
            anchor_box_gt[:, 0] - anchors[:, 0]) / anchors[:, 2]
        anchor_box_pos_u_targets = (
            anchor_box_gt[:, 1] - anchors[:, 1]) / anchors[:, 3]

        anchor_box_pos_v_targets *= 10.0
        anchor_box_pos_u_targets *= 10.0

        anchor_box_dim_h_targets = tf.math.log(
            anchor_box_gt[:, 2] / anchors[:, 2])
        anchor_box_dim_w_targets = tf.math.log(
            anchor_box_gt[:, 3] / anchors[:, 3])

        anchor_box_dim_h_targets *= 5.0
        anchor_box_dim_w_targets *= 5.0

        anchor_box_targets = tf.stack([anchor_box_pos_v_targets,
                                       anchor_box_pos_u_targets,
                                       anchor_box_dim_h_targets,
                                       anchor_box_dim_w_targets], axis=1)

        class_targets = tf.gather(gt_classes, max_ious, axis=0)

        num_classes = tf.shape(gt_classes)[1]
        negative_label = tf.one_hot(
            num_classes - 1, num_classes, on_value=1.0, off_value=0.0)

        negative_targets = tf.ones([tf.shape(anchors)[0], 1]) * negative_label

        positive_anchor_mask = tf.broadcast_to(tf.expand_dims(positive_anchor_mask, -1), tf.shape(class_targets))

        anchor_class_targets = tf.where(
            positive_anchor_mask, class_targets, negative_targets)

        return anchor_box_targets, anchor_class_targets


def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.name_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor
