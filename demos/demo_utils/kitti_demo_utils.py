import numpy as np
import warnings

import src.core.constants as constants
import src.retina_net.datasets.dataset_utils as dataset_utils


def read_labels(
    label_path,
    difficulty='hard',
    categories=(
        'car',
        'pedestrian',
        'cyclist')):
    """
    Reads ground truth labels and parses them into one hot class representation and groundtruth 2D bounding box.

    """
    # Extract the list
    labels = np.loadtxt(label_path,
                        delimiter=' ',
                        dtype=str,
                        usecols=np.arange(start=0, step=1, stop=15))

    if len(labels.shape) == 1:
        labels = np.array([labels.tolist()])

    # Filter labels
    diff_dict = constants.KITTI_DIFF_DICTS[difficulty.lower()]
    min_height = diff_dict['min_height']
    max_truncation = diff_dict['max_truncation']
    max_occlusion = diff_dict['max_occlusion']

    heights = labels[:, 7].astype(
        np.float32) - labels[:, 5].astype(np.float32)

    class_filter = np.asarray(
        [inst.lower() in categories for inst in labels[:, 0]])

    height_filter = heights >= min_height
    truncation_filter = labels[:, 1].astype(np.float) <= max_truncation
    occlusion_filter = labels[:, 2].astype(np.float) <= max_occlusion

    final_filter = class_filter & height_filter & truncation_filter & occlusion_filter

    labels = np.array(labels[final_filter])

    boxes_2d_gt = dataset_utils.kitti_labels_to_boxes_2d(labels)
    boxes_class_gt = []

    if boxes_2d_gt.size == 0:
        boxes_2d_gt = np.array([])
        boxes_class_gt.append([])
    else:
        for elem in labels[:, 0]:
            if elem.lower() == 'car':
                boxes_class_gt.append([1, 0, 0, 0])
            elif elem.lower() == 'pedestrian' or elem.lower() == 'person_sitting':
                boxes_class_gt.append([0, 1, 0, 0])
            elif elem.lower() == 'cyclist':
                boxes_class_gt.append([0, 0, 1, 0])

    if len(boxes_2d_gt.shape) == 1:
        boxes_2d_gt = np.expand_dims(boxes_2d_gt, axis=0)

    return np.array(boxes_class_gt).astype(
        np.float32), np.array(boxes_2d_gt).astype(
        np.float32)


def read_predictions(
    prediction_path,
    categories=(
        'car',
        'pedestrian',
        'cyclist',
        'dontcare')):
    """
    Reads predictions from text file

    """
    # Extract the list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = np.loadtxt(prediction_path,
                                 delimiter=' ',
                                 dtype=str,
                                 usecols=np.arange(start=0, step=1, stop=16))

    if len(predictions) == 0:
        return np.array([]), np.array([]), np.array([])

    if len(predictions.shape) == 1:
        predictions = np.array([predictions.tolist()])

    # Filter labels
    class_filter = np.asarray(
        [inst.lower() in categories for inst in predictions[:, 0]])

    predictions = np.array(predictions[class_filter])

    boxes_2d = dataset_utils.kitti_labels_to_boxes_2d(predictions)
    boxes_score = predictions[:, 15]
    boxes_class = []

    if boxes_2d.size == 0:
        boxes_2d = np.array([])
        boxes_class.append([])
    else:
        for elem in predictions[:, 0]:
            if elem.lower() == 'car':
                boxes_class.append([1, 0, 0, 0])
            elif elem.lower() == 'pedestrian':
                boxes_class.append([0, 1, 0, 0])
            elif elem.lower() == 'cyclist':
                boxes_class.append([0, 0, 1, 0])
            else:
                boxes_class.append([0, 0, 0, 1])

    if len(boxes_2d.shape) == 1:
        boxes_2d = np.expand_dims(boxes_2d, axis=0)
        boxes_score = np.expand_dims(boxes_score, axis=0)

    return np.array(boxes_class).astype(
        np.float32), np.array(boxes_2d).astype(
        np.float32), np.array(boxes_score).astype(
        np.float32)


def read_labels_tracking(
    label_path,
    num_elems,
    difficulty='hard',
    categories=(
        'car',
        'pedestrian',
        'cyclist')):
    """
    Reads ground truth labels and parses them into one hot class representation and groundtruth 2D bounding box.

    """
    # Extract the list
    labels = np.loadtxt(label_path,
                        delimiter=' ',
                        dtype=str,
                        usecols=np.arange(start=0, step=1, stop=15))

    if len(labels.shape) == 1:
        labels = np.array([labels.tolist()])

    # Filter labels
    diff_dict = constants.KITTI_DIFF_DICTS[difficulty.lower()]
    min_height = diff_dict['min_height']
    max_truncation = diff_dict['max_truncation']
    max_occlusion = diff_dict['max_occlusion']

    heights = labels[:, 9].astype(
        np.float32) - labels[:, 7].astype(np.float32)

    class_filter = np.asarray(
        [inst.lower() in categories for inst in labels[:, 2]])

    height_filter = heights >= min_height
    truncation_filter = labels[:, 3].astype(np.float) <= max_truncation
    occlusion_filter = labels[:, 4].astype(np.float) <= max_occlusion

    final_filter = class_filter & height_filter & truncation_filter & occlusion_filter

    labels = np.array(labels[final_filter])

    frame_dict = dict()

    for i in range(num_elems):
        frame_fitler = labels[:, 0].astype(np.float) == i
        frame_labels = labels[frame_fitler]

        obj_id = frame_labels[:, 1].astype(np.float)
        boxes_class_gt = frame_labels[:, 2]
        boxes_2d_gt = dataset_utils.kitti_labels_to_boxes_2d(
            frame_labels[:, 2:])

        frame_dict.update({i: {'boxes_2d_gt': boxes_2d_gt,
                               'boxes_class_gt': boxes_class_gt,
                               'obj_id': obj_id}})

    return frame_dict
