import numpy as np
import tensorflow as tf


def vuhw_to_vuvu(vuhw):
    """
    Function to transform between center_dimensions box representation (vuhw) to
    top left and bottom right corner box representation (vuvu).

    :param vuhw: N x 4 tensor representing the v, u, width and height of bounding boxes
    :return: vuvu: N x 4 tensor represeting v_min, u_min, v_max, u_max, the corners of the bounding boxes.
    """
    v = vuhw[:, 0]
    u = vuhw[:, 1]
    h = vuhw[:, 2]
    w = vuhw[:, 3]

    v_min = v - h / 2.0
    u_min = u - w / 2.0
    v_max = v + h / 2.0
    u_max = u + w / 2.0

    return tf.stack((v_min, u_min, v_max, u_max), axis=1)


def vuvu_to_vuhw(vuvu):
    """
    Function to transform between top left and bottom right corner box representation (vuvu) to
     center_dimensions box representation (vuhw) .

    :param vuvu: N x 4 tensor represeting v_min, u_min, v_max, u_max, the corners of the bounding boxes.
    :return: vuhw: N x 4 tensor representing the v, u, width and height of bounding boxes
    """

    v_min = vuvu[:, 0]
    u_min = vuvu[:, 1]
    v_max = vuvu[:, 2]
    u_max = vuvu[:, 3]

    v = (v_max + v_min) / 2.0
    u = (u_max + u_min) / 2.0
    h = v_max - v_min
    w = u_max - u_min

    return tf.stack((v, u, h, w), axis=1)


def vuvu_to_vuhw_np(vuvu):
    """
    Function to transform between top left and bottom right corner box representation (vuvu) to
     center_dimensions box representation (vuhw) .

    :param vuvu: N x 4 tensor represeting v_min, u_min, v_max, u_max, the corners of the bounding boxes.
    :return: vuhw: N x 4 tensor representing the v, u, width and height of bounding boxes
    """

    v_min = vuvu[:, 0]
    u_min = vuvu[:, 1]
    v_max = vuvu[:, 2]
    u_max = vuvu[:, 3]

    v = (v_max + v_min) / 2.0
    u = (u_max + u_min) / 2.0
    h = v_max - v_min
    w = u_max - u_min

    return np.stack((v, u, h, w), axis=1)


def vuhw_to_vuvu_np(vuhw):
    """
    Function to transform between center_dimensions box representation (vuhw) to
    top left and bottom right corner box representation (vuvu).

    :param vuhw: N x 4 tensor representing the v, u, width and height of bounding boxes
    :return: vuvu: N x 4 tensor representing v_min, u_min, v_max, u_max, the corners of the bounding boxes.
    """
    v = vuhw[:, 0]
    u = vuhw[:, 1]
    h = vuhw[:, 2]
    w = vuhw[:, 3]

    v_min = v - h / 2.0
    u_min = u - w / 2.0
    v_max = v + h / 2.0
    u_max = u + w / 2.0

    return np.stack((v_min, u_min, v_max, u_max), axis=1)


def crop_vuvu_to_image_shape(vuvu, im_shape):
    """
    Crop bounding boxes with (vuvu) representation to fit in image space.

    :param vuvu: N x 4 tensor represeting v_min, u_min, v_max, u_max, the corners of the bounding boxes.
    :param im_shape: Shape of image as [batch, height, width, depth]

    :return: Cropped 2D bounding box
    """
    im_shape = tf.cast(im_shape, tf.float32)
    im_height = im_shape[1]
    im_width = im_shape[2]

    v_min = vuvu[:, 0]
    u_min = vuvu[:, 1]
    v_max = vuvu[:, 2]
    u_max = vuvu[:, 3]

    v_min = tf.minimum(im_height - 1, tf.maximum(0.0, v_min))
    u_min = tf.minimum(im_width - 1, tf.maximum(0.0, u_min))
    v_max = tf.minimum(im_height - 1, tf.maximum(0.0, v_max))
    u_max = tf.minimum(im_width - 1, tf.maximum(0.0, u_max))

    return tf.stack((v_min, u_min, v_max, u_max), axis=1)


def bbox_iou_vuvu(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with v1, u1, v2, u2  point order.
        bboxes2: shape (total_bboxes2, 4)
            with v1, u1, v2, u2  point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    y11, x11, y12, x12 = tf.split(bboxes1, 4, axis=1)
    y21, x21, y22, x22 = tf.split(bboxes2, 4, axis=1)
    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))
    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))
    inter_area = tf.maximum((xI2 - xI1 + 1.0), 0.0) * \
        tf.maximum((yI2 - yI1 + 1.0), 0.0)
    bboxes1_area = (x11 - x12 + 1.0) * (y11 - y12 + 1.0)
    bboxes2_area = (x21 - x22 + 1.0) * (y21 - y22 + 1.0)
    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area
    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union + 0.00001)


def box_from_anchor_and_target(anchors, regressed_targets):
    """
    Get bounding box from anchor and target through transformation provided in the paper.
    :param anchors: Nx4 anchor boxes
    :param regressed_targets: Nx4 regression targets
    :return:
    """

    boxes_v = anchors[:, 2] * regressed_targets[:, 0] / 10.0 + anchors[:, 0]
    boxes_u = anchors[:, 3] * regressed_targets[:, 1] / 10.0 + anchors[:, 1]

    boxes_h = anchors[:, 2] * \
        tf.clip_by_value(tf.exp(regressed_targets[:, 2] / 5.0), 1e-4, 1e4)
    boxes_w = anchors[:, 3] * \
        tf.clip_by_value(tf.exp(regressed_targets[:, 3] / 5.0), 1e-4, 1e4)

    return tf.stack([boxes_v,
                     boxes_u,
                     boxes_h,
                     boxes_w], axis=1)


def box_from_anchor_and_target_bnms(anchors, regressed_targets):
    """
    Batch version of the above function. For usage with MC-Dropout.

    :param anchors: MxNx4 anchor boxes where M is the batch size.
    :param regressed_targets: MxNx4 regression targets
    :return:
    """
    boxes_v = anchors[:, :, 2] * \
        regressed_targets[:, :, 0] / 10.0 + anchors[:, :, 0]
    boxes_u = anchors[:, :, 3] * \
        regressed_targets[:, :, 1] / 10.0 + anchors[:, :, 1]

    boxes_h = anchors[:, :, 2] * \
        tf.clip_by_value(tf.exp(regressed_targets[:, :, 2] / 5.0), 1e-4, 1e4)
    boxes_w = anchors[:, :, 3] * \
        tf.clip_by_value(tf.exp(regressed_targets[:, :, 3] / 5.0), 1e-4, 1e4)

    return tf.stack([boxes_v,
                     boxes_u,
                     boxes_h,
                     boxes_w], axis=2)


def normalize_2d_bounding_boxes(boxes_2, shape):
    """
    Produces bounding boxes as a fraction of the total image shape. Useful for crop-and-resize
    :param boxes_2: 2D bounding box
    :param shape: Image shape
    :return: Normalized 2D bounding box
    """
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    normalizer = tf.stack([height, width, height, width], axis=0)
    return boxes_2 / normalizer


def expand_2d_bounding_boxes(boxes_2, shape):
    """
    Produces bounding boxes by multiplying normalized boxes with new image shape.
     Usefull for handling image resize

    :param boxes_2: 2D bounding box
    :param shape: Image shape
    :return: scaled 2D bounding box
    """
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    scale = tf.stack([height, width, height, width], axis=0)
    return boxes_2 * scale
