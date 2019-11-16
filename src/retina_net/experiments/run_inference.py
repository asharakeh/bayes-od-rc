import argparse
import os
import time
import sys
import json

import yaml
import tensorflow as tf
import numpy as np

import src.core as core
import src.retina_net.experiments.validation_utils as val_utils

from src.retina_net import config_utils
from src.retina_net.builders import dataset_handler_builder
from src.retina_net.models.retinanet_model import RetinaNetModel
from src.retina_net.anchor_generator import box_utils
from src.retina_net.experiments import inference_utils

keras = tf.keras


def test_model(config):
    # Get testing config
    test_config = config['testing_config']
    ckpt_idx = test_config['ckpt_idx']
    uncertainty_method = test_config['uncertainty_method']
    use_full_covar = test_config['use_full_covar']
    nms_config = test_config['nms_config']

    # Create dataset class
    dataset_config = config['dataset_config']
    training_dataset = dataset_config['dataset']
    dataset_config['dataset'] = test_config['test_dataset']
    dataset_handler = dataset_handler_builder.build_dataset(
        dataset_config, 'test')

    # Set keras training phase
    keras.backend.set_learning_phase(0)
    print("Keras Learning Phase Set to: " +
          str(keras.backend.learning_phase()))

    # Create Model
    with tf.name_scope("retinanet_model"):
        model = RetinaNetModel(config['model_config'])

    # Initialize the model from a saved checkpoint
    checkpoint_dir = os.path.join(
        core.data_dir(), 'outputs',
        config['checkpoint_name'], 'checkpoints', config['checkpoint_name'])
    predictions_dir = os.path.join(
        core.data_dir(), 'outputs',
        config['checkpoint_name'], 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    if not os.path.exists(checkpoint_dir):
        raise ValueError('{} must have at least one checkpoint entry.'
                         .format(checkpoint_dir))

    # Instantiate mini-batch and epoch size
    epoch_size = int(dataset_handler.epoch_size)

    # Create Dataset
    # Main function to create dataset
    dataset = dataset_handler.create_dataset()

    # Batch size goes in parenthesis.
    batched_dataset = dataset.repeat(1).batch(1)

    # `prefetch` lets the dataset fetch batches, in the background while the model is validating.
    batched_dataset = batched_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    print('Starting inference at ' +
          time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

    # Initialize the model checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=model)

    # Begin inference loop
    all_checkpoint_states = tf.train.get_checkpoint_state(
        checkpoint_dir).all_model_checkpoint_paths

    start = time.time()

    checkpoint_to_restore = all_checkpoint_states[ckpt_idx - 1]
    ckpt_id = val_utils.strip_checkpoint_id(checkpoint_to_restore)

    # Make directories if these dont exist
    predictions_dir_ckpt = os.path.join(predictions_dir,
                                        'testing',
                                        dataset_config['dataset'],
                                        str(ckpt_id),
                                        uncertainty_method)
    if dataset_config['dataset'] == 'rvc':
        predictions_dir_ckpt = os.path.join(
            predictions_dir_ckpt,
            dataset_config['rvc']['paths_config']['sequence_dir'])

    if uncertainty_method == 'bayes_od':
        predictions_dir_ckpt += '_' + \
            test_config['bayes_od_config']['fusion_method']

    os.makedirs(os.path.join(predictions_dir_ckpt, 'data'), exist_ok=True)

    loc_mean_dir = os.path.join(predictions_dir_ckpt, 'mean')
    loc_cov_dir = os.path.join(predictions_dir_ckpt, 'cov')

    cat_param_dir = os.path.join(predictions_dir_ckpt, 'cat_param')
    cat_count_dir = os.path.join(predictions_dir_ckpt, 'cat_count')

    os.makedirs(loc_mean_dir, exist_ok=True)
    os.makedirs(loc_cov_dir, exist_ok=True)
    os.makedirs(cat_param_dir, exist_ok=True)
    os.makedirs(cat_count_dir, exist_ok=True)
    print('\nRunning checkpoint ' + str(ckpt_id) + '\n')

    # Restore checkpoint. expect_partial is needed to get rid of
    # optimizer/loss graph elements.
    ckpt.restore(checkpoint_to_restore).expect_partial()

    # Perform dataset-specific setup of result output
    if dataset_config['dataset'] == 'kitti':
        pass
    elif dataset_config['dataset'] == 'bdd' or dataset_config['dataset'] == 'coco' or dataset_config['dataset'] == 'pascal':
        final_results_list = []
        # Single json file for bdd dataset
        prediction_json_file_name = os.path.join(
            predictions_dir_ckpt, 'data', 'predictions.json')
    elif dataset_config['dataset'] == 'rvc':
        final_results_list = []
        # Single json file for bdd dataset
        prediction_json_file_name = os.path.join(
            predictions_dir_ckpt, 'data', 'predictions.json')

    # Inference loop starts here. Iterate over samples once.
    for counter, sample_dict in enumerate(batched_dataset):
        output_class_counts, output_boxes_vuhw, output_covs, nms_indices, predicted_boxes_iou_mat = inference_utils.bayes_od_inference(
            model, sample_dict, test_config['bayes_od_config'], nms_config, dataset_name=dataset_config['dataset'], use_full_covar=use_full_covar)

        output_class_counts = output_class_counts.numpy()
        output_boxes_vuhw = output_boxes_vuhw.numpy()
        output_covs = output_covs.numpy()
        nms_indices = nms_indices.numpy()
        predicted_boxes_iou_mat = predicted_boxes_iou_mat.numpy()

        if output_boxes_vuhw.size > 0:
            output_classes, output_boxes_vuhw, output_covs, output_counts = inference_utils.bayes_od_clustering(
                output_class_counts, output_boxes_vuhw, output_covs, nms_indices, predicted_boxes_iou_mat, affinity_threshold=nms_config['iou_threshold'])
            if output_boxes_vuhw.size > 0:
                output_boxes_vuhw = np.squeeze(output_boxes_vuhw, axis=2)
                output_boxes = box_utils.vuhw_to_vuvu_np(output_boxes_vuhw)
            else:
                output_boxes_vuhw = output_boxes_vuhw
                output_boxes = output_boxes_vuhw
                output_counts = output_boxes_vuhw
        else:
            output_classes = output_boxes_vuhw
            output_boxes = output_boxes_vuhw
            output_covs = output_boxes_vuhw
            output_counts = output_boxes_vuhw

        # Perform index mapping in case training and testing datasets are not
        # the same
        if training_dataset != dataset_config['dataset'] and output_boxes.size > 0:
            if dataset_config['dataset'] == 'kitti_tracking':
                dataset_config['dataset'] = 'kitti'
            output_classes_mapped = inference_utils.map_dataset_classes(
                training_dataset, dataset_config['dataset'], output_classes)
        else:
            output_classes_mapped = output_classes

        # Perform dataset-specific saving of outputs
        if dataset_config['dataset'] == 'kitti':
            predictions_kitti_format = val_utils.predictions_to_kitti_format(
                output_boxes, output_classes_mapped)

            prediction_file_name = os.path.join(
                predictions_dir_ckpt,
                'data',
                dataset_handler.sample_ids[counter] + '.txt')

            mean_file_name = os.path.join(
                loc_mean_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            covar_file_name = os.path.join(
                loc_cov_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            cat_param_file_name = os.path.join(
                cat_param_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            cat_count_file_name = os.path.join(
                cat_count_dir, dataset_handler.sample_ids[counter] + '.npy')

            if predictions_kitti_format.size == 0:
                np.savetxt(prediction_file_name, [])
            else:
                np.savetxt(
                    prediction_file_name,
                    predictions_kitti_format,
                    newline='\r\n',
                    fmt='%s')

        elif dataset_config['dataset'] == 'bdd':
            predictions_bdd_format = val_utils.predictions_to_bdd_format(
                output_boxes,
                output_classes_mapped,
                dataset_handler.sample_ids[counter],
                category_list=dataset_handler.training_data_config['categories'])
            final_results_list.extend(predictions_bdd_format)

            mean_file_name = os.path.join(
                loc_mean_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            covar_file_name = os.path.join(
                loc_cov_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            cat_param_file_name = os.path.join(
                cat_param_dir,
                dataset_handler.sample_ids[counter] + '.npy')

            cat_count_file_name = os.path.join(
                cat_count_dir, dataset_handler.sample_ids[counter] + '.npy')

        sys.stdout.write(
            '\r{}'.format(counter + 1) + ' /' + str(epoch_size))

        np.save(mean_file_name, output_boxes_vuhw)
        np.save(covar_file_name, output_covs)
        np.save(cat_param_file_name, output_classes)
        np.save(cat_count_file_name, output_counts)

    elapsed_time = time.time() - start
    time_per_sample = elapsed_time / dataset_handler.epoch_size
    frame_rate = 1.0 / time_per_sample

    print("\nMean frame rate: " + str(frame_rate))

    # Final dataset-specific wrap up work for checkpoint
    # results
    if dataset_config['dataset'] == 'kitti':
        pass
    else:
        with open(prediction_json_file_name, 'w') as fp:
            json.dump(final_results_list, fp, indent=4,
                      separators=(',', ': '))


def main():
    """Object Detection Model Validator
    """

    # Defaults
    default_gpu_device = '0'
    default_config_path = core.model_dir(
        'retina_net') + '/configs/retinanet_bdd.yaml'
    # Allowed data splits are 'train','train_mini', 'val', 'val_half',
    # 'val_mini'
    default_data_split = 'val'

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

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load in configuration file as python dictionary
    with open(args.yaml_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make necessary directories, update config with checkpoint path and data
    # split
    config = config_utils.setup(config, args)

    # Go to inference function
    test_model(config)


if __name__ == '__main__':
    main()
