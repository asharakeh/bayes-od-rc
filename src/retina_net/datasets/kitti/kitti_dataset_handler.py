import csv
import glob
import os
import random

import numpy as np
import tensorflow as tf

import src.core.constants as constants
import src.retina_net.anchor_generator.box_utils as box_utils
import src.retina_net.datasets.dataset_utils as dataset_utils

from src.core.abstract_classes.dataset_handler import DatasetHandler
from src.retina_net.anchor_generator.fpn_anchor_generator import FpnAnchorGenerator


class KittiDatasetHandler(DatasetHandler):
    def __init__(self, config, train_val_test):
        """
        Initializes directories, and loads the sample list

        :param config: configuration dictionary
        :param train_val_test (string): 'train', 'val', or 'test'
        """
        super().__init__(config)

        # Define Dicts
        self._MEANS_DICT = constants.MEANS_DICT
        self._DIFF_DICTS = constants.KITTI_DIFF_DICTS
        self.resize_shape = config['kitti']['resize_shape']

        # Load configs
        self.training_data_config = config['kitti']['training_data_config']
        self.anchor_gen_config = config['anchor_generator']

        # Define paths to dataset
        paths_config = config['kitti']['paths_config']
        self.dataset_dir = os.path.expanduser(paths_config['dataset_dir'])
        self.data_split_dir = paths_config['data_split_dir']

        self.im_dir = os.path.join(
            self.dataset_dir, self.data_split_dir, 'image_2')
        self.gt_label_dir = os.path.join(
            self.dataset_dir, self.data_split_dir, 'label_2')

        # Make sure dataset directories exist
        dataset_utils.check_data_dirs([self.im_dir, self.gt_label_dir])

        # Make sure training-validation data split exists
        txt_files = glob.glob(os.path.expanduser(self.dataset_dir) + '/*.txt')
        possible_splits = [os.path.splitext(os.path.basename(path))[
            0] for path in txt_files]
        if self.data_split not in possible_splits:
            raise ValueError(
                'Invalid dataset_split: {}. Possible splits include: {}'.format(
                    self.data_split, possible_splits))

        # Load sample paths for the chosen split
        self.sample_ids = self._load_sample_ids()

        # Random shuffle here is much more computationally efficient than
        # randomly shuffling a dataset iterator.
        if train_val_test == 'train':
            random.shuffle(self.sample_ids)

        self.epoch_size = len(self.sample_ids)
        self._create_sample_paths(self.sample_ids)

        # Create placeholder for dataset
        self.dataset = None

        # Create flag if train\val or just inference
        self.is_testing = (train_val_test == 'test')

    def set_paths(self, sample):
        """
        Function to set sample paths, in the case of usage for a demo

        :param sample: kitti sample name
        :return: None
        """
        self.im_paths = [self.im_dir + '/' + sample + '.png']
        self.label_paths = [self.gt_label_dir + '/' + sample + '.txt']

    def create_dataset(self):
        """
        Create dataset using tf.dataset API

        :return: dataset : dataset object
        """
        # Set data path lists
        im_paths = self.im_paths
        label_paths = self.label_paths

        # Create dataset using API
        dataset = tf.data.Dataset.from_tensor_slices((im_paths,
                                                      label_paths))

        # Create sample dictionary
        self.dataset = dataset.map(
            self.create_sample_dict,
            num_parallel_calls=10)

        return self.dataset

    @tf.function
    def create_sample_dict(
            self,
            im_path,
            label_path):
        """
        Creates sample dictionary for a single sample

        :param im_path: left image path
        :param label_path: ground truth label path

        :return: sample_dict: Sample dictionary filled with input tensors
        """
        with tf.name_scope('input_data'):
            # Read left and right images
            image_as_string = tf.io.read_file(im_path)
            image = tf.image.decode_png(image_as_string, channels=3)

            # Resize images
            image_resized = tf.image.resize(
                image,
                self.resize_shape,
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=True)

            image_resized = tf.image.resize_with_crop_or_pad(
                image_resized, self.resize_shape[0], self.resize_shape[1])

            boxes_class_gt, boxes_2d_gt, no_gt = tf.py_function(
                self._read_labels, [label_path], [
                    tf.float32, tf.float32, tf.bool])

            # Rescale 2D GT bounding boxes to accommodate image resize
            boxes_2d_norm = box_utils.normalize_2d_bounding_boxes(
                boxes_2d_gt, tf.shape(image))
            boxes_2d_gt = box_utils.expand_2d_bounding_boxes(
                boxes_2d_norm, tf.shape(image_resized))

            # Normalize Images In Preparation For Feature Extractor
            image_norm = dataset_utils.mean_image_subtraction(
                image_resized, self._MEANS_DICT[self.im_normalization])

            # Flip channels to BGR since pretrained weights use this
            # configuration.
            channels = tf.unstack(image_norm, axis=-1)
            image_norm = tf.stack(
                [channels[2], channels[1], channels[0]], axis=-1)

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
                        ious, self.anchor_gen_config['min_positive_iou'], self.anchor_gen_config['max_negative_iou'])

                    anchors_positive_mask_list.append(positive_anchor_mask)
                    anchors_negative_mask_list.append(negative_anchor_mask)

                    anchor_box_targets, anchor_class_targets = generator.generate_anchor_targets(
                        anchors, boxes_2d_gt_vuhw, boxes_class_gt, max_ious, positive_anchor_mask)
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

    def _load_sample_ids(self):
        """
        Load the sample ids listed in the data split file (e.g. train.txt, val.txt, test.txt, train_mini.txt ..)

        :return: paths: A list of sample ids read from the .txt file corresponding to the data split
        """
        paths = []
        split_file = os.path.join(self.dataset_dir, self.data_split + '.txt')
        with open(split_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=' ')
            for row in csv_reader:
                for sample in row:
                    paths.append(sample)

        return np.array(paths)

    def _create_sample_paths(self, sample_ids):
        """
        Creates sample paths
        """
        self.im_paths = [
            self.im_dir +
            '/' +
            sample +
            '.png' for sample in sample_ids]

        self.label_paths = [self.gt_label_dir + '/' +
                            sample + '.txt' for sample in sample_ids]

    def _read_labels(self, label_path):
        """
        Reads ground truth labels and parses them into one hot class representation and groundtruth 2D bounding box.
        """
        label_path = label_path.numpy()

        # Extract the list
        no_gt = False

        difficulty = self.training_data_config['difficulty']
        categories = self.training_data_config['categories']

        label_path = label_path.decode("utf-8")

        labels = np.loadtxt(label_path,
                            delimiter=' ',
                            dtype=str,
                            usecols=np.arange(start=0, step=1, stop=15))

        if len(labels.shape) == 1:
            labels = np.array([labels.tolist()])

        # Filter labels
        diff_dict = self._DIFF_DICTS[difficulty.lower()]
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
            boxes_2d_gt = np.array([0.0, 0.0, 1.0, 1.0])
            boxes_class_gt.append([0, 0, 0, 1])
            no_gt = True
        else:
            for elem in labels[:, 0]:
                if elem.lower() == 'car':
                    boxes_class_gt.append([1, 0, 0, 0])
                elif elem.lower() == 'pedestrian':
                    boxes_class_gt.append([0, 1, 0, 0])
                elif elem.lower() == 'cyclist':
                    boxes_class_gt.append([0, 0, 1, 0])

        if len(boxes_2d_gt.shape) == 1:
            boxes_2d_gt = np.expand_dims(boxes_2d_gt, axis=0)

        return [np.array(boxes_class_gt).astype(np.float32),
                np.array(boxes_2d_gt).astype(np.float32),
                no_gt]
