import os
import json
import random

import numpy as np
import tensorflow as tf

import src.core.constants as constants
import src.retina_net.anchor_generator.box_utils as box_utils
import src.retina_net.datasets.dataset_utils as dataset_utils

from src.core.abstract_classes.dataset_handler import DatasetHandler
from src.retina_net.anchor_generator.fpn_anchor_generator import FpnAnchorGenerator


class BddDatasetHandler(DatasetHandler):
    def __init__(self, config, train_val_test):
        """
        Initializes directories, and loads the sample list

        :param config: configuration dictionary
        :param train_val_test (string): 'train', 'val', or 'test'
        """
        super().__init__(config)

        # Define Dicts
        self._MEANS_DICT = constants.MEANS_DICT

        # Load configs
        self.anchor_gen_config = config['anchor_generator']
        self.training_data_config = config['bdd']['training_data_config']
        # Define paths to dataset
        paths_config = config['bdd']['paths_config']
        self.dataset_dir = os.path.expanduser(paths_config['dataset_dir'])
        data_set_size = paths_config['100k_or_10k']

        if train_val_test == 'train':
            self.data_split_dir = train_val_test
            self.label_file_name = 'train.json'
            self.frac_training_data = self.training_data_config['frac_training_data']
        else:
            self.data_split_dir = 'val'
            self.label_file_name = 'val.json'
            self.frac_training_data = 1.0

        self.im_dir = os.path.join(
            self.dataset_dir, 'images', data_set_size, self.data_split_dir)
        self.gt_label_dir = os.path.join(
            self.dataset_dir, 'labels')

        # Make sure dataset directories exist
        dataset_utils.check_data_dirs([self.im_dir, self.gt_label_dir])

        # Get sample ids
        self._load_sample_ids()
        self.epoch_size = len(self.sample_ids)
        self.labels = json.load(open(os.path.join(
            self.gt_label_dir, self.label_file_name), 'r'))

        # Create placeholder for dataset
        self.dataset = None

        # Create flag if train\val or just inference
        self.is_testing = (train_val_test == 'test')

    def _load_sample_ids(self):
        """
        loads sample ids to read dataset
        """
        sample_ids = os.listdir(self.im_dir)

        # Random shuffle here is much more computationally efficient than randomly shuffling a dataset iterator.
        if self.frac_training_data != 1.0 and self.data_split_dir == 'train':
            percent_samples = int(len(sample_ids) * self.frac_training_data)
            inds = np.random.choice(
                len(sample_ids), percent_samples, replace=False)
            self.sample_ids = [sample_ids[ind] for ind in inds]
        elif self.data_split_dir == 'train':
            random.shuffle(sample_ids)
            self.sample_ids = sample_ids
        else:
            self.sample_ids = sample_ids

        # Create a list of image paths from sample ids
        self.im_paths = [
            self.im_dir +
            '/' +
            sample for sample in self.sample_ids]

    def set_sample_id(self, sample_index):
        self.im_paths = [self.im_paths[sample_index]]
        self.sample_ids = [self.sample_ids[sample_index]]

    def create_dataset(self):
        """
        Create dataset using tf.dataset API

        :return: dataset : dataset object
        """
        # Set data path lists
        im_paths = self.im_paths
        sample_ids = self.sample_ids

        # Create dataset using API
        dataset = tf.data.Dataset.from_tensor_slices((im_paths, sample_ids))

        # Create sample dictionary
        self.dataset = dataset.map(
            self.create_sample_dict,
            num_parallel_calls=10)

        return self.dataset

    def create_sample_dict(
            self,
            im_path,
            sample_id):
        """
        Creates sample dictionary for a single sample

        :param im_path: left image path
        :param sample_id: ground truth sample id

        :return: sample_dict: Sample dictionary filled with input tensors
        """
        with tf.name_scope('input_data'):
            # Read image
            image_as_string = tf.io.read_file(im_path)
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = tf.cast(image, tf.float32)

            image_norm = dataset_utils.mean_image_subtraction(
                image, self._MEANS_DICT[self.im_normalization])

            # Flip channels to BGR since pretrained weights use this
            # configuration.
            channels = tf.unstack(image_norm, axis=-1)
            image_norm = tf.stack(
                [channels[2], channels[1], channels[0]], axis=-1)

            boxes_class_gt, boxes_2d_gt, no_gt = tf.py_function(
                self._read_labels, [sample_id], [
                    tf.float32, tf.float32, tf.bool])

        # Create_sample_dict
        sample_dict = dict()
        sample_dict.update({constants.IMAGE_NORMALIZED_KEY: image_norm})
        sample_dict.update(
            {constants.ORIGINAL_IM_SIZE_KEY: tf.shape(image)})

        # Create prior anchors and anchor targets
        generator = FpnAnchorGenerator(self.anchor_gen_config)
        boxes_2d_gt_vuhw = box_utils.vuvu_to_vuhw(boxes_2d_gt)

        anchors_list = []
        anchors_class_target_list = []
        anchors_box_target_list = []
        anchors_positive_mask_list = []
        anchors_negative_mask_list = []

        for layer_number in self.anchor_gen_config['layers']:
            anchors = generator.generate_anchors(
                tf.shape(image_norm), layer_number)
            anchors_list.append(anchors)

            if not self.is_testing:
                anchor_corners = box_utils.vuhw_to_vuvu(anchors)
                ious = box_utils.bbox_iou_vuvu(anchor_corners, boxes_2d_gt)

                positive_anchor_mask, negative_anchor_mask, max_ious = generator.positive_negative_batching(
                    ious, self.anchor_gen_config['min_positive_iou'],
                    self.anchor_gen_config['max_negative_iou'])

                anchors_positive_mask_list.append(positive_anchor_mask)
                anchors_negative_mask_list.append(negative_anchor_mask)

                anchor_box_targets, anchor_class_targets = generator.generate_anchor_targets(
                    anchors, boxes_2d_gt_vuhw, boxes_class_gt, max_ious,
                    positive_anchor_mask)
                anchors_box_target_list.append(anchor_box_targets)
                anchors_class_target_list.append(anchor_class_targets)

        # Sample dict is stacked from p3 --> p7, this is essential to
        # memorize for stacking the predictions later on
        sample_dict.update(
            {constants.ANCHORS_KEY: tf.concat(anchors_list, axis=0)})
        if not self.is_testing:
            sample_dict.update({constants.ANCHORS_BOX_TARGETS_KEY: tf.concat(
                anchors_box_target_list, axis=0),
                constants.ANCHORS_CLASS_TARGETS_KEY: tf.concat(
                anchors_class_target_list, axis=0),
                constants.POSITIVE_ANCHORS_MASK_KEY: tf.concat(
                anchors_positive_mask_list, axis=0),
                constants.NEGATIVE_ANCHOR_MASK_KEY: tf.concat(
                anchors_negative_mask_list, axis=0)})

        return sample_dict

    def _read_labels(self, sample_id):
        """
        Reads ground truth labels and parses them into one hot class representation and groundtruth 2D bounding box.
        """
        sample_id = sample_id.numpy()

        # Extract the list
        no_gt = False
        categories = self.training_data_config['categories']
        boxes_class_gt = []

        sample_id = sample_id.decode("utf-8")

        frame_labels = [label for label in self.labels if
                        label['name'] == sample_id and label[
                            'category'] in categories]

        boxes_2d_gt = np.array([[label['bbox'][1],
                                 label['bbox'][0],
                                 label['bbox'][3],
                                 label['bbox'][2]] for label in frame_labels])

        categories_gt = [label['category'] for label in frame_labels]

        if boxes_2d_gt.size == 0:
            cat_one_hot = [0 for e in range(len(categories) + 1)]
            boxes_2d_gt = np.array([0.0, 0.0, 1.0, 1.0])
            boxes_class_gt.append(cat_one_hot)
            no_gt = True
        else:
            for elem in categories_gt:
                cat_one_hot = [0 for e in range(len(categories) + 1)]

                cat_idx = categories.index(elem.lower())
                cat_one_hot[cat_idx] = 1
                boxes_class_gt.append(cat_one_hot)

        # one-hot representation dependent on config file

        if len(boxes_2d_gt.shape) == 1:
            boxes_2d_gt = np.expand_dims(boxes_2d_gt, axis=0)
        return [np.array(boxes_class_gt).astype(np.float32),
                np.array(boxes_2d_gt).astype(np.float32),
                no_gt]
