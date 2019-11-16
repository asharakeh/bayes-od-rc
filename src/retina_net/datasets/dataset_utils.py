import os

import numpy as np
import tensorflow as tf


def check_data_dirs(folders):
    """Checks that the dataset directories exist in the file system
    :param folders:
    :return:  FileNotFoundError: if a folder is missing
    """

    for folder in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(
                'Folder does not exist: {}'.format(folder))


def mean_image_subtraction(image, channel_means):
    """
    Normalizes image by subtracting image mean.

    :param image: image in RGB format
    :param channel_means: channel_means dict
    :return: normalized image
    """
    channel_means = tf.constant(channel_means, dtype=tf.float32)
    channel_means = tf.reshape(channel_means, [1, 1, 3])
    return image - channel_means


############################
# KITTI specific functions #
############################
def kitti_labels_to_boxes_2d(labels):
    """
    Transforms labels to 2D box format.

    Args:
        labels: input kitti_obj labels (see kitti_obj utils)


    Returns:
        boxes_2d: n x 4 array representing 2d boxes represented as
        [y1, x1, y2, x2]
    """
    boxes_2d = []

    if len(labels.shape) < 2:
        labels = np.array([labels.tolist()])

    for label in labels:
        x1 = float(label[4])
        y1 = float(label[5])
        x2 = float(label[6])
        y2 = float(label[7])
        box_2d = [y1, x1, y2, x2]
        boxes_2d.append(box_2d)

    return np.array(boxes_2d)
