import argparse
import os
import time
import datetime

import yaml
import tensorflow as tf
import numpy as np

import src.core as core
from src.retina_net import config_utils
from src.core import constants
from src.retina_net.builders import dataset_handler_builder
from src.retina_net.models.retinanet_model import RetinaNetModel

keras = tf.keras


def train_model(config):
    """
    Training function.
    :param config: config file
    """
    # Get training config
    training_config = config['training_config']
    # Create dataset class
    dataset_config = config['dataset_config']
    dataset_handler = dataset_handler_builder.build_dataset(
        dataset_config, 'train')

    # Set keras training phase
    keras.backend.set_learning_phase(1)
    print("Keras Learning Phase Set to: " +
          str(keras.backend.learning_phase()))

    # Create Model
    with tf.name_scope("retinanet_model"):
        model = RetinaNetModel(config['model_config'])

    # Instantiate an optimizer.
    minibatch_size = training_config['minibatch_size']

    epoch_size = int(dataset_handler.epoch_size / minibatch_size)

    initial_learning_rate = training_config['initial_learning_rate']
    decay_factor = training_config['decay_factor']

    decay_boundaries = [
        boundary *
        epoch_size for boundary in training_config['decay_boundaries']]
    decay_factors = [decay_factor**i for i in range(0, len(decay_boundaries)+1)]
    learning_rate_values = [
        np.round(
            initial_learning_rate *
            decay_factor,
            8) for decay_factor in decay_factors]

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        decay_boundaries, learning_rate_values)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-2)

    # Create summary writer
    log_file = config['logs_dir'] + '/training/' + str(datetime.datetime.now())
    summary_writer = tf.summary.create_file_writer(log_file)

    # Load checkpoint weights if training folder exists
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0),
        optimizer=optimizer,
        net=model)
    manager = tf.train.CheckpointManager(
        ckpt,
        config['checkpoint_path'],
        max_to_keep=training_config['max_checkpoints_to_keep'])

    ckpt.restore(manager.latest_checkpoint)

    # If no checkpoints exist, intialize either from imagenet or from scratch
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        ckpt.step.assign_add(1)
    elif config['model_config']['feature_extractor']['pretrained_initialization']:
        # Load resnet-50 imagenet pretrained weights if set in config file.
        # Dummy input required to define graph.
        input_shape = (224, 224, 3)
        dummy_input = keras.layers.Input(shape=input_shape)
        model.feature_extractor(dummy_input)

        weights_path = keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            ('https://github.com/fchollet/deep-learning-models/'
             'releases/download/v0.2/'
             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'),
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        model.feature_extractor.load_weights(weights_path, by_name=True)
        # Tensorflow 2.0 bug with loading weights in nested models. Might get
        # fixed later.
        model.feature_extractor.conv_block_2a.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.conv_block_3a.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.conv_block_4a.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.conv_block_5a.load_weights(
            weights_path, by_name=True)

        model.feature_extractor.identity_block_2b.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_2c.load_weights(
            weights_path, by_name=True)

        model.feature_extractor.identity_block_3b.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_3c.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_3d.load_weights(
            weights_path, by_name=True)

        model.feature_extractor.identity_block_4b.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_4c.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_4d.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_4e.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_4f.load_weights(
            weights_path, by_name=True)

        model.feature_extractor.identity_block_5b.load_weights(
            weights_path, by_name=True)
        model.feature_extractor.identity_block_5c.load_weights(
            weights_path, by_name=True)
        print("Initializing from ImageNet weights.")
    else:
        print("Initializing from scratch.")

    # Create Dataset
    # Skip already passed elements in dataset, in case of resuming training.
    dataset = dataset_handler.create_dataset().repeat(
        training_config['max_epochs'])

    # Batch size goes in parenthesis.
    batched_dataset = dataset.batch(minibatch_size)

    batched_dataset = batched_dataset.take(tf.data.experimental.cardinality(
        batched_dataset) - tf.cast(ckpt.step + 1, tf.int64))
    print("Remaining iterations:" +
          str(tf.data.experimental.cardinality(batched_dataset).numpy()))

    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    batched_dataset = batched_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    last_time = time.time()
    for sample_dict in batched_dataset:
        with summary_writer.as_default():
            # Turn on both graph and profiler for debugging the graph in
            # tensorboard
            tf.summary.trace_on(graph=False, profiler=False)
            total_loss, loss_dict = train_single_step(
                model, optimizer, sample_dict)
            tf.summary.trace_export(
                name="training_trace",
                step=0,
                profiler_outdir=log_file)

            with tf.name_scope('losses'):
                for loss_name in loss_dict.keys():
                    tf.summary.scalar(loss_name,
                                      loss_dict[loss_name],
                                      step=int(ckpt.step))
            with tf.name_scope('optimizer'):
                tf.summary.scalar('learning_rate',
                                  lr_schedule(int(ckpt.step)),
                                  step=int(ckpt.step))

            tf.summary.scalar(
                'Total Loss',
                total_loss,
                step=int(
                    ckpt.step))

            summary_writer.flush()

            # Write summary
            if int(ckpt.step) % training_config['summary_interval'] == 0:
                current_time = time.time()
                time_elapsed = current_time - last_time
                last_time = time.time()
                print(
                    'Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                        int(ckpt.step), total_loss.numpy(), time_elapsed))

            # Saving checkpoint
            if int(ckpt.step) % int(
                    epoch_size * training_config['checkpoint_interval']) == 0:
                save_path = manager.save(checkpoint_number=ckpt.save_counter)
                print("Saved checkpoint for step {}: {}".format(
                    int(ckpt.step), save_path))
                print("loss {:1.2f}".format(total_loss.numpy()))
                ckpt.step.assign_add(1)
            else:
                ckpt.step.assign_add(1)

@tf.function
def train_single_step(
        model,
        optimizer,
        sample_dict):
    """
    :param model: keras retinanet model
    :param optimizer: keras optimizer
    :param sample_dict: input dictionary generated from dataset.
    If element sizes in this dictionary are variable, remove tf.function decorator.

    :return total_loss: Sum of all losses.
    :return cls_loss: classification loss.
    :return reg_loss: regression loss.
    :return regularization_loss: regularization_loss
    :return prediction_dict: Dictionary containing neural network predictions
    """
    with tf.GradientTape() as tape:
        prediction_dict = model(sample_dict[constants.IMAGE_NORMALIZED_KEY],
                                train_val_test='training')

        total_loss, loss_dict = model.get_loss(sample_dict, prediction_dict)

        # Get any regularization loss in the model and add it to total loss
        regularization_loss = tf.reduce_sum(
            tf.concat([layer.losses for layer in model.layers], axis=0))
        loss_dict.update(
            {constants.REGULARIZATION_LOSS_KEY: regularization_loss})

        total_loss += regularization_loss

        # Compute the gradient which respect to the loss
    with tf.name_scope("grad_ops"):
        gradients = tape.gradient(total_loss, model.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        optimizer.apply_gradients(
            zip(clipped_gradients, model.trainable_variables))

    return total_loss, loss_dict


def main():
    """Object Detection Model Trainer
    """

    # Defaults
    default_gpu_device = '1'
    default_config_path = core.model_dir(
        'retina_net') + '/configs/retinanet_bdd.yaml'
    # Allowed data splits are 'train','train_mini', 'val', 'val_half',
    # 'val_mini'
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

    # Allow GPU memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load in configuration file as python dictionary
    with open(args.yaml_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Make necessary directories, update config with checkpoint path and data
    # split
    config = config_utils.setup(config, args)

    # Go to training function
    train_model(config)


if __name__ == '__main__':
    main()
