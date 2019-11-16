import argparse
import os
import time
import datetime
import sys
import json

import yaml
import tensorflow as tf
import numpy as np

import src.core as core
import src.retina_net.experiments.validation_utils as val_utils

from src.retina_net import config_utils
from src.core import constants
from src.retina_net.builders import dataset_handler_builder
from src.retina_net.models.retinanet_model import RetinaNetModel

keras = tf.keras


def validate_model(config):
    # Get validation config
    val_config = config['validation_config']
    eval_wait_interval = val_config['eval_wait_interval']

    # Create dataset class
    dataset_config = config['dataset_config']
    dataset_handler = dataset_handler_builder.build_dataset(
        dataset_config, 'val')

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

    already_evaluated_ckpts = val_utils.get_evaluated_ckpts(predictions_dir)

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

    log_file = config['logs_dir'] + \
        '/validation/' + str(datetime.datetime.now())
    summary_writer = tf.summary.create_file_writer(log_file)

    print('Starting evaluation at ' +
          time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

    # Repeated checkpoint run
    last_checkpoint_id = -1
    number_of_evaluations = 0

    # Initialize the model checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=model)

    # Begin validation loop
    while True:
        all_checkpoint_states = tf.train.get_checkpoint_state(
            checkpoint_dir).all_model_checkpoint_paths
        num_checkpoints = len(all_checkpoint_states)
        print("Total Checkpoints: ", num_checkpoints)
        start = time.time()
        if number_of_evaluations >= num_checkpoints:
            print('\nNo new checkpoints found in %s. '
                  'Will try again in %d seconds.'
                  % (checkpoint_dir, eval_wait_interval))
        else:
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = all_checkpoint_states[ckpt_idx]
                ckpt_id = val_utils.strip_checkpoint_id(checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id or ckpt_id == 1:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                # run_checkpoint_once
                predictions_dir_ckpt = os.path.join(predictions_dir,
                                                    'validation',
                                                    str(ckpt_id),
                                                    'data')

                os.makedirs(predictions_dir_ckpt, exist_ok=True)
                print('\nRunning checkpoint ' + str(ckpt_id) + '\n')

                ckpt.restore(checkpoint_to_restore)

                # Perform dataset-specific setup of result output
                if dataset_config['dataset'] == 'kitti':
                    pass
                elif dataset_config['dataset'] == 'bdd' or dataset_config['dataset'] == 'coco':
                    final_results_list = []
                    # Single json file for bdd or coco dataset
                    prediction_json_file_name = os.path.join(
                        predictions_dir_ckpt, 'predictions.json')
                elif dataset_config['dataset'] == 'rvc':
                    final_results_list = []
                    # Single json file for bdd dataset
                    predictions_dir_ckpt = os.path.join(
                        predictions_dir_ckpt, dataset_config['rvc']['paths_config']['sequence_dir'])
                    os.makedirs(predictions_dir_ckpt, exist_ok=True)
                    prediction_json_file_name = os.path.join(
                        predictions_dir_ckpt,
                        'predictions.json')

                with summary_writer.as_default():
                    for counter, sample_dict in enumerate(batched_dataset):

                        total_loss, loss_dict, prediction_dict = val_single_step(
                            model, sample_dict)

                        output_classes, output_boxes = val_utils.post_process_predictions(
                            sample_dict, prediction_dict, dataset_name=dataset_config['dataset'])

                        output_boxes = output_boxes.numpy()
                        output_classes = output_classes.numpy()

                        # Perform dataset-specific saving of outputs
                        if dataset_config['dataset'] == 'kitti':
                            predictions_kitti_format = val_utils.predictions_to_kitti_format(
                                output_boxes, output_classes)

                            prediction_file_name = os.path.join(
                                predictions_dir_ckpt,
                                dataset_handler.sample_ids[counter] + '.txt')

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
                                output_classes,
                                dataset_handler.sample_ids[counter],
                                category_list=dataset_handler.training_data_config['categories'])
                            final_results_list.extend(predictions_bdd_format)

                        elif dataset_config['dataset'] == 'coco':
                            predictions_coco_format = val_utils.predictions_to_coco_format(
                                output_boxes,
                                output_classes,
                                int(dataset_handler.sample_ids[counter][:-4]),
                                dataset_handler.training_data_to_coco_category_ids)
                            final_results_list.extend(predictions_coco_format)

                        elif dataset_config['dataset'] == 'rvc':
                            predictions_rvc_format = val_utils.predictions_to_rvc_format(
                                output_boxes,
                                output_classes,
                                dataset_handler.sample_ids[counter][:-4],
                                dataset_handler.training_data_categories)
                            final_results_list.extend(predictions_rvc_format)

                        with tf.name_scope('losses'):
                            for loss_name in loss_dict.keys():
                                tf.summary.scalar(loss_name,
                                                  loss_dict[loss_name],
                                                  step=int(ckpt.step))

                        tf.summary.scalar(
                            'Total Loss',
                            total_loss,
                            step=int(
                                ckpt.step))

                        summary_writer.flush()

                        sys.stdout.write(
                            '\r{}'.format(
                                counter +
                                1) +
                            ' /' +
                            str(epoch_size))

                    # Final dataset-specific wrap up work for checkpoint
                    # results
                    if dataset_config['dataset'] == 'kitti':
                        pass
                    else:
                        with open(prediction_json_file_name, 'w') as fp:
                            json.dump(final_results_list, fp, indent=4,
                                      separators=(',', ': '))

                    number_of_evaluations += 1
                    val_utils.write_evaluated_ckpts(
                        predictions_dir, np.array([ckpt_id]))
                    # Save the id of the latest evaluated checkpoint
                    last_checkpoint_id = ckpt_id

        time_to_next_eval = start + eval_wait_interval - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)


@tf.function
def val_single_step(
        model,
        sample_dict):
    """
    :param model: keras retinanet model
    :param sample_dict: input dictionary generated from dataset.
    If element sizes in this dictionary are variable, remove tf.function decorator.

    :return total_loss: Sum of all losses.
    :return cls_loss: classification loss.
    :return reg_loss: regression loss.
    :return regularization_loss: regularization_loss
    :return prediction_dict: Dictionary containing neural network predictions
    """

    prediction_dict = model(sample_dict[constants.IMAGE_NORMALIZED_KEY],
                            train_val_test='validation')

    total_loss, loss_dict = model.get_loss(sample_dict, prediction_dict)

    # Get any regularization loss in the model and add it to total loss
    regularization_loss = tf.reduce_sum(
        tf.concat([layer.losses for layer in model.layers], axis=0))
    loss_dict.update(
        {constants.REGULARIZATION_LOSS_KEY: regularization_loss})

    total_loss += regularization_loss

    return total_loss, loss_dict, prediction_dict


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

    # Allow GPU memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load in configuration file as python dictionary
    with open(args.yaml_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make necessary directories, update config with checkpoint path and data
    # split
    config = config_utils.setup(config, args)

    # Go to validation function
    validate_model(config)


if __name__ == '__main__':
    main()
