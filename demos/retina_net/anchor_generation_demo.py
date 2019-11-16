import argparse
import os
import random
import yaml

import numpy as np
import tensorflow as tf
import cv2

import src.core as core
import src.core.constants as constants
import src.retina_net.anchor_generator.box_utils as box_utils

from src.retina_net import config_utils
from src.retina_net.builders import dataset_handler_builder
from demos.demo_utils.vis_utils import draw_box_2d


def main():
    """Anchor Generator Demo
    """

    # Defaults
    default_gpu_device = '0'
    dataset_name = 'bdd'
    default_config_path = core.model_dir(
        'retina_net') + '/configs/retinanet_' + dataset_name + '.yaml'

    default_data_split = 'train'

    # Parse input
    parser = argparse.ArgumentParser()  # Define argparser object
    parser.add_argument('--gpu_device',
                        type=str,
                        dest='gpu_device',
                        default=default_gpu_device)

    parser.add_argument('--yaml_path',
                        type=str,
                        dest='yaml_path',
                        default=default_config_path)

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split)
    args = parser.parse_args()

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device

    # Load in configuration file as python dictionary
    with open(args.yaml_path, 'r') as yaml_file:
        config_raw = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make necessary directories, update config with checkpoint path and data
    # split
    config = config_utils.setup(config_raw, args)

    # Create dataset class
    dataset_config = config['dataset_config']
    dataset_handler = dataset_handler_builder.build_dataset(
        dataset_config, 'train')

    # Set which frame to show
    if dataset_name == 'kitti':
        frames_list = os.listdir(dataset_handler.gt_label_dir)
        index = random.randint(0, len(frames_list))
        frame_id = frames_list[index][0:6]
        print('Showing Frame: ' + frame_id)
        dataset_handler.set_paths(frame_id)
    elif dataset_name == 'bdd':
        frames_list = os.listdir(dataset_handler.im_dir)
        index = random.randint(0, len(frames_list))
        frame_id = frames_list[index]
        print('Showing Frame: ' + frame_id)
        dataset_handler.set_sample_id(index)

    dataset = dataset_handler.create_dataset()

    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    dataset = dataset.repeat()
    dataset = dataset.batch(1)

    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ############################
    # Get anchors and GT boxes #
    ############################
    for sample_dict in dataset.take(1):
        anchors = sample_dict[constants.ANCHORS_KEY][0]

        positive_mask = sample_dict[constants.POSITIVE_ANCHORS_MASK_KEY][0]

        num_positives = tf.reduce_sum(tf.cast(positive_mask, tf.float32))

        anchor_box_targets = sample_dict[constants.ANCHORS_BOX_TARGETS_KEY][0]

        anchor_class_targets = sample_dict[constants.ANCHORS_CLASS_TARGETS_KEY][0]

        # Reconstruct anchors from gt box targets
        gt_2d_boxes = box_utils.box_from_anchor_and_target(
            anchors, anchor_box_targets)

        positive_anchor_inds = tf.where(positive_mask)
        positive_anchors = tf.gather(anchors, positive_anchor_inds, axis=0)
        positive_anchors = tf.reshape(positive_anchors, [-1, 4])
        positive_anchors_corners = box_utils.vuhw_to_vuvu(positive_anchors)

        gt_2d_boxes_corner = box_utils.vuhw_to_vuvu(gt_2d_boxes)

        gt_2d_boxes_corner = tf.gather(
            gt_2d_boxes_corner, positive_anchor_inds, axis=0)
        gt_classes = tf.gather(
            anchor_class_targets,
            positive_anchor_inds,
            axis=0)

        gt_2d_boxes_corner = tf.reshape(gt_2d_boxes_corner, [-1, 4])
        gt_classes = tf.reshape(gt_classes, [-1, tf.shape(gt_classes)[2]])

        print('Number of Positive Anchors: ' + str(num_positives.numpy()))

        image_normalized = np.squeeze(
            sample_dict[constants.IMAGE_NORMALIZED_KEY])

        rgb_means = constants.MEANS_DICT['ImageNet']

        image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB) + rgb_means

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        #########################################
        # Create Images and Draw Bounding Boxes #
        #########################################
        image_out = draw_box_2d(np.copy(image),
                                positive_anchors_corners.numpy(),
                                line_width=2,
                                is_gt=False)

        image_out = draw_box_2d(image_out,
                                gt_2d_boxes_corner.numpy(),
                                gt_classes.numpy(),
                                line_width=2,
                                is_gt=True)

        cv2.imshow('Positive Anchors and Ground Truth', image_out)
        cv2.waitKey()


if __name__ == '__main__':
    main()
