import os
import tensorflow as tf
import numpy as np

import src.core.constants as constants

from src.retina_net.anchor_generator import box_utils


@tf.function
def post_process_predictions(sample_dict, prediction_dict, dataset_name='bdd'):
    """
    Prepares predictions to be written into file. Also applies non-maximum suppression.
    :param sample_dict: input dictionary generated from dataset.
    If element sizes in this dictionary are variable, remove tf.function decorator.

    :param prediction_dict: Dictionary containing neural network predictions
    :param dataset_name: Name of dataset, to be used to apply dataset specific post processing.
    :return:
    """

    anchors = sample_dict[constants.ANCHORS_KEY][0]
    predicted_box_targets = prediction_dict[constants.ANCHORS_BOX_PREDICTIONS_KEY][0]

    predicted_boxes = box_utils.box_from_anchor_and_target(
        anchors, predicted_box_targets)

    predicted_boxes_corners = box_utils.vuhw_to_vuvu(predicted_boxes)
    predicted_boxes_classes = tf.nn.softmax(
        prediction_dict[constants.ANCHORS_CLASS_PREDICTIONS_KEY][0], axis=1)

    num_classes = tf.cast(tf.shape(predicted_boxes_classes)[1], tf.int64)

    category_filter = tf.not_equal(
        tf.argmax(
            predicted_boxes_classes,
            axis=1),
        num_classes - 1)

    predicted_boxes_corners = tf.boolean_mask(predicted_boxes_corners,
                                              category_filter)
    predicted_boxes_classes = tf.boolean_mask(predicted_boxes_classes,
                                              category_filter)

    top_scores = tf.reduce_max(predicted_boxes_classes, axis=1)

    nms_indices, _ = tf.image.non_max_suppression_with_scores(
        predicted_boxes_corners,
        top_scores,
        max_output_size=100,
        iou_threshold=0.5,
        soft_nms_sigma=0.5)

    if dataset_name == 'kitti':
        predicted_boxes_corners_normalized = box_utils.normalize_2d_bounding_boxes(
            predicted_boxes_corners,
            tf.shape(sample_dict[constants.IMAGE_NORMALIZED_KEY][0]))
        predicted_boxes_corners_scaled = box_utils.expand_2d_bounding_boxes(
            predicted_boxes_corners_normalized, sample_dict[constants.ORIGINAL_IM_SIZE_KEY][0])
    elif dataset_name == 'coco':
        predicted_boxes_corners_shifted = predicted_boxes_corners - \
            sample_dict[constants.IMAGE_PADDING_KEY][0]
        predicted_boxes_corners_normalized = box_utils.normalize_2d_bounding_boxes(predicted_boxes_corners_shifted, tf.shape(
            sample_dict[constants.IMAGE_NORMALIZED_KEY][0])[0:2] - tf.cast(2 * sample_dict[constants.IMAGE_PADDING_KEY][0, 0:2], tf.int32))
        predicted_boxes_corners_scaled = box_utils.expand_2d_bounding_boxes(
            predicted_boxes_corners_normalized, sample_dict[constants.ORIGINAL_IM_SIZE_KEY][0])
    else:
        predicted_boxes_corners_scaled = predicted_boxes_corners

    predicted_boxes_classes_out = tf.gather(predicted_boxes_classes,
                                            nms_indices,
                                            axis=0)
    predicted_boxes_corners_out = tf.gather(predicted_boxes_corners_scaled,
                                            nms_indices,
                                            axis=0)

    return predicted_boxes_classes_out, predicted_boxes_corners_out


def get_evaluated_ckpts(prediction_dir):
    already_evaluated_ckpts = []
    file_path = prediction_dir + '/evaluated_ckpts.txt'

    if os.path.exists(file_path):
        evaluated_ckpts = np.loadtxt(file_path, dtype=np.int)
        if np.isscalar(evaluated_ckpts):
            # one entry
            already_evaluated_ckpts = np.asarray(
                [evaluated_ckpts], np.int32)
        else:
            already_evaluated_ckpts = np.asarray(evaluated_ckpts,
                                                 np.int32)
    return already_evaluated_ckpts


def strip_checkpoint_id(checkpoint_dir):
    """Helper function to return the checkpoint index number.

    Args:
        checkpoint_dir: Path directory of the checkpoints

    Returns:
        checkpoint_id: An int representing the checkpoint index
    """

    checkpoint_name = checkpoint_dir.split('-')[-1]
    return int(checkpoint_name)


def write_evaluated_ckpts(predictions_dir, ckpt_name):
    file_path = os.path.join(predictions_dir, 'evaluated_ckpts.txt')
    with open(file_path, 'ba') as fp:
        np.savetxt(fp, ckpt_name, fmt='%d')
    return


def predictions_to_rvc_format(
        output_boxes,
        output_classes,
        image_id,
        training_data_categories):
    """
    Writes detections in coco format

    :param output_boxes: output bounding boxes
    :param output_classes: output class scores

    :return: Array of predictions ready to be written to file.
    """
    bdd_boxes_list = []
    for box_2d, box_class, in zip(output_boxes, output_classes):

        bdd_box_dict = dict()

        max_ind = np.argmax(box_class)

        if max_ind < (output_classes.shape[1] - 1):
            bdd_box_dict.update({"image_id": image_id,
                                 "category_id": training_data_categories[max_ind],
                                 "bbox": [float(box_2d[0]),
                                          float(box_2d[1]),
                                          float(box_2d[2]),
                                          float(box_2d[3])],
                                 "score": box_class[max_ind].tolist()})
            bdd_boxes_list.append(bdd_box_dict)

    return bdd_boxes_list


def predictions_to_coco_format(
        output_boxes,
        output_classes,
        image_id,
        training_data_to_coco_category_ids):
    """
    Writes detections in coco format

    :param output_boxes: output bounding boxes
    :param output_classes: output class scores

    :return: Array of predictions ready to be written to file.
    """
    bdd_boxes_list = []
    for box_2d, box_class, in zip(output_boxes, output_classes):

        bdd_box_dict = dict()

        max_ind = np.argmax(box_class)

        if max_ind < (output_classes.shape[1] - 1):
            bdd_box_dict.update({"image_id": image_id,
                                 "category_id": training_data_to_coco_category_ids[max_ind],
                                 "bbox": [float(box_2d[1]),
                                          float(box_2d[0]),
                                          float(box_2d[3] - box_2d[1]),
                                          float(box_2d[2] - box_2d[0])],
                                 "score": box_class[max_ind].tolist()})
            bdd_boxes_list.append(bdd_box_dict)

    return bdd_boxes_list


def predictions_to_bdd_format(
        output_boxes,
        output_classes,
        frame_name,
        category_list):
    """
    Writes detections in BDD format

    :param output_boxes: output bounding boxes
    :param output_classes: output classes

    :return: Array of predictions ready to be written to file.
    """
    bdd_boxes_list = []
    for box_2d, box_class, in zip(output_boxes, output_classes):

        bdd_box_dict = dict()

        max_ind = np.argmax(box_class)

        if max_ind < len(category_list):
            bdd_box_dict.update({"name": frame_name,
                                 "timestep": 1000,
                                 "category": category_list[max_ind],
                                 "bbox": [float(box_2d[1]),
                                          float(box_2d[0]),
                                          float(box_2d[3]),
                                          float(box_2d[2])],
                                 "score": box_class[max_ind].tolist()})
            bdd_boxes_list.append(bdd_box_dict)

    return bdd_boxes_list


def predictions_to_kitti_format(output_boxes, output_classes):
    """
    Writes detections in KITTI format

    :param output_boxes: output bounding boxes
    :param output_classes: output classes

    :return: Array of predictions ready to be written to file.
    """
    kitti_boxes_array = []

    for box_2d, box_class, in zip(output_boxes, output_classes):

        kitti_box_array = []
        box_2d = np.copy(box_2d[::-1])

        max_ind = np.argmax(box_class)

        if max_ind == 0:
            kitti_box_array.extend(['Car', -1, -1, -10])
            kitti_box_array.extend(box_2d[2:4])
            kitti_box_array.extend(box_2d[0:2])
            kitti_box_array.extend([-10, -10, -10])
            kitti_box_array.extend([-10, -10, -10])
            kitti_box_array.extend([-10])
            kitti_box_array.append(box_class[max_ind])
            kitti_boxes_array.append(kitti_box_array)
        elif max_ind == 1:
            kitti_box_array.extend(['Pedestrian', -1, -1, -10])
            kitti_box_array.extend(box_2d[2:4])
            kitti_box_array.extend(box_2d[0:2])
            kitti_box_array.extend([-10, -10, -10])
            kitti_box_array.extend([-10, -10, -10])
            kitti_box_array.extend([-10])
            kitti_box_array.append(box_class[max_ind])
            kitti_boxes_array.append(kitti_box_array)
        # elif max_ind == 2:
        #    kitti_box_array.extend(['Cyclist', -1, -1, -10])
        #    kitti_box_array.extend(box_2d[2:4])
        #    kitti_box_array.extend(box_2d[0:2])
        #    kitti_box_array.extend([-10, -10, -10])
        #    kitti_box_array.extend([-10, -10, -10])
        #    kitti_box_array.extend([-10])
        #    kitti_box_array.append(box_class[max_ind])
        #    kitti_boxes_array.append(kitti_box_array)
        # else:
        #    kitti_box_array.extend(['DontCare', -1, -1, -10])
        #    kitti_box_array.extend(box_2d[2:4])
        #    kitti_box_array.extend(box_2d[0:2])
        #    kitti_box_array.extend([-10, -10, -10])
        #    kitti_box_array.extend([-10, -10, -10])
        #    kitti_box_array.extend([-10])
        #    kitti_box_array.append(box_class[max_ind])
        #    kitti_boxes_array.append(kitti_box_array)

    return np.asarray(kitti_boxes_array)
