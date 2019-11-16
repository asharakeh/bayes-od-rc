import os
import random

import numpy as np
import cv2
import json

import core

from core.evaluation_utils_2d import two_d_iou
from demos.demo_utils.vis_utils import draw_box_2d
from demos.demo_utils.bdd_demo_utils import read_bdd_format


def main():
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    data_split_dir = 'val'

    # Specify whether the validation or inference results need to be
    # visualized.
    # results_dir = 'validation'  # Or testing
    results_dir = 'testing'

    # sample_free, anchor_redundancy, black_box,  bayes_od_none,
    # bayes_od_ci_fast. bayes_od_ci,or bayes_od_ici
    uncertainty_method = 'bayes_od_ci_fast'

    dataset_dir = os.path.expanduser('~/Datasets/bdd100k')
    image_dir = os.path.join(dataset_dir, 'images', '100k', data_split_dir)
    label_file_name = os.path.join(
        dataset_dir, 'labels', data_split_dir) + '.json'

    checkpoint_name = 'retinanet_bdd'
    checkpoint_number = '101'

    if results_dir == 'testing':
        prediction_dir = os.path.join(
            core.data_dir(),
            'outputs',
            checkpoint_name,
            'predictions',
            results_dir,
            'bdd',
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
    prediction_file_name = os.path.join(prediction_dir, 'predictions.json')

    frames_list = os.listdir(image_dir)
    index = random.randint(0, len(frames_list))
    frame_id = frames_list[index]

    print('Showing Frame ID:' + frame_id)

    #############
    # Read Frame
    #############
    im_path = image_dir + '/' + frame_id
    image = cv2.imread(im_path)

    all_labels = json.load(open(label_file_name, 'r'))
    category_gt, boxes_2d_gt, _ = read_bdd_format(
        frame_id, all_labels, categories=[
            'car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor'])

    all_predictions = json.load(open(prediction_file_name, 'r'))
    category_pred, boxes_2d_pred, _ = read_bdd_format(
        frame_id, all_predictions, categories=[
            'car', 'truck', 'bus', 'person', 'rider', 'bike', 'motor'])

    max_ious = np.zeros(boxes_2d_pred.shape[0])
    # Compute IOU between each prediction and the ground truth boxes
    if boxes_2d_gt.size > 0 and boxes_2d_pred.size > 0:
        for obj_idx in range(boxes_2d_pred.shape[0]):
            obj_iou_fmt = boxes_2d_pred[obj_idx]

            ious_2d = two_d_iou(obj_iou_fmt, boxes_2d_gt)

            max_iou = np.amax(ious_2d)
            max_ious[obj_idx] = max_iou

    #########################################################
    # Draw GT and Prediction Boxes
    #########################################################
    # Transform Predictions to left and right images
    image_out = draw_box_2d(image,
                            boxes_2d_gt,
                            category_gt,
                            line_width=2,
                            dataset='bdd',
                            is_gt=True)

    image_out = draw_box_2d(image_out,
                            boxes_2d_pred,
                            category_pred,
                            line_width=2,
                            is_gt=False,
                            dataset='bdd',
                            text_to_plot=max_ious,
                            plot_text=True)

    if results_dir == 'testing':
        cv2.imshow('Detections from ' + uncertainty_method, image_out)
    else:
        cv2.imshow('Validation Set Detections', image_out)

    cv2.waitKey()


if __name__ == "__main__":
    main()
