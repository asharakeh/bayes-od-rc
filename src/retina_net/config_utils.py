import os
import shutil

import src.core as core


def _check_config_name(config, yaml_path):
    # Check if config and checkpoint name match
    checkpoint_name = config['checkpoint_name']
    config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    if config_name != checkpoint_name:
        raise ValueError('Config and checkpoint names must match.')


def _add_ckpt_paths(config):
    """Adds new entries to the config for the checkpoint path
        checkpoint_dir - output checkpoint directory
        checkpoint_path - output checkpoint path
    """
    checkpoint_name = config['checkpoint_name']

    # Make checkpoint directories
    config['checkpoint_dir'] = os.path.join(
        core.data_dir(),
        'outputs',
        config['checkpoint_name'],
        'checkpoints')
    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])

    # Determine checkpoint path
    config['checkpoint_path'] = \
        config['checkpoint_dir'] + '/' + checkpoint_name


def _setup_output_dir(config, yaml_path):
    """Creates the output directory and copies the config there
    """
    checkpoint_name = config['checkpoint_name']

    # Make log directories
    logdir = os.path.join(core.data_dir(), 'outputs',
                          checkpoint_name, 'logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config['logs_dir'] = logdir
    # Save config to experiments folder for bookkeeping
    experiment_dir = os.path.join(core.data_dir(), 'outputs',
                                  checkpoint_name, checkpoint_name + '.yaml')
    shutil.copy(yaml_path, experiment_dir)


def setup(config, args):
    """Ensures config and checkpoint names match and creates the log and
    checkpoint directories.

    Args:
        config: Configuration dictionary
        args: args object from an experiment script containing data_split and yaml_path


    Returns:
        config: Configuration dictionary with new entries:
            checkpoint_dir - output checkpoint directory
            checkpoint_path - output checkpoint path
            dataset_config/data_split - dataset data split
    """

    # Check that config name matches checkpoint name
    _check_config_name(config, args.yaml_path)

    # Add checkpoint path to the config
    _add_ckpt_paths(config)

    # Add data split to dataset config
    # RetinaNet Specific
    dataset_name = config['dataset_config']['dataset']
    num_classes = len(config['dataset_config'][dataset_name]['training_data_config']['categories'])

    num_scales = len(config['dataset_config']['anchor_generator']['scales'])
    num_aspect_ratios = len(config['dataset_config']['anchor_generator']['aspect_ratios'])

    anchors_per_location = num_scales * num_aspect_ratios

    config['model_config']['header']['num_classes'] = num_classes
    config['model_config']['header']['anchors_per_location'] = anchors_per_location

    config['dataset_config']['num_classes'] = num_classes

    config['dataset_config']['data_split'] = args.data_split

    # Setup the output directory
    _setup_output_dir(config, args.yaml_path)

    return config
