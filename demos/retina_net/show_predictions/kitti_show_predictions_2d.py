import os
import random

import numpy as np
import cv2

import core

from core.evaluation_utils_2d import two_d_iou
from demos.demo_utils.vis_utils import draw_box_2d

from demos.demo_utils.kitti_demo_utils import read_labels, read_predictions


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'training'

    # Specify whether the validation or inference results need to be
    # visualized.
    #results_dir = 'validation'
    results_dir = 'testing'

    # sample_free, anchor_redundancy, black_box,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od'

    dataset_dir = os.path.expanduser('~/Datasets/Kitti/object/')
    image_dir = os.path.join(dataset_dir, data_split_dir) + '/image_2'
    label_dir = os.path.join(dataset_dir, data_split_dir) + '/label_2'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    if results_dir == 'testing':
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            'kitti',
            checkpoint_number,
            uncertainty_method,
            'data')
    else:
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            checkpoint_number,
            'data')

    frames_list = os.listdir(prediction_dir)
    index = random.randint(0, len(frames_list) - 1)
    frame_id = int(frames_list[index][0:6])

    # frame_id = 27  # Out of distribution example
    frame_id = 4079
    # frame_id = 169   # Many Cars, Hard
    # frame_id = 2290  # Many Cars, Occlusions
    # frame_id = 1941  # Many Cars, Horizontal Direction
    # frame_id = 4032  # Orientation
    # frame_id = 104   # Weird Orientation
    # frame_id = 7047  # Weird Orientation
    # frame_id = 6632 # Very hard orientations

    # frame_id = 195  # Single Pedestrian
    # frame_id = 1574  # Single Pedestrian
    # frame_id = 332  # Multiple Hard Pedestrians
    # frame_id = 1193 # Multiple Hard Pedestrians

    # frame_id = 1274 # Multiple Cyclists

    print('Showing Frame: %d' % frame_id)

    #############
    # Read Frame
    #############
    im_path = image_dir + '/{:06d}.png'.format(frame_id)
    image = cv2.imread(im_path)

    label_path = label_dir + '/{:06d}.txt'.format(frame_id)
    gt_classes_hard, gt_boxes_hard = read_labels(label_path)

    prediction_path = prediction_dir + '/{:06d}.txt'.format(frame_id)
    prediction_classes, prediction_boxes, prediction_scores = read_predictions(
        prediction_path)

    max_ious = np.zeros(prediction_boxes.shape[0])
    # Compute IOU between each prediction and the ground truth boxes
    if gt_boxes_hard.size > 0 and prediction_boxes.size > 0:
        for obj_idx in range(prediction_boxes.shape[0]):
            obj_iou_fmt = prediction_boxes[obj_idx]

            ious_2d = two_d_iou(obj_iou_fmt, gt_boxes_hard)

            max_iou = np.amax(ious_2d)
            max_ious[obj_idx] = max_iou
    #########################################################
    # Draw GT and Prediction Boxes
    #########################################################
    # Transform Predictions to left and right images
    image_out = draw_box_2d(image,
                            gt_boxes_hard,
                            gt_classes_hard,
                            line_width=2,
                            dataset='kitti',
                            is_gt=True)

    image_out = draw_box_2d(image_out,
                            prediction_boxes,
                            prediction_classes,
                            line_width=2,
                            is_gt=False,
                            dataset='kitti',
                            text_to_plot=max_ious,
                            plot_text=True)
    if results_dir == 'testing':
        cv2.imshow('Detections from ' + uncertainty_method, image_out)
    else:
        cv2.imshow('Validation Set Detections', image_out)

    cv2.waitKey()


if __name__ == "__main__":
    main()
