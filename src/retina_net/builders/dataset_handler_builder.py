from src.retina_net.datasets.kitti.kitti_dataset_handler import KittiDatasetHandler
from src.retina_net.datasets.bdd.bdd_dataset_handler import BddDatasetHandler


def build_dataset(dataset_config, train_val_test):
    """
    Builds dataset handler based on which dataset is currently being used.

    :param dataset_config: dataset configuration dict.
    :param train_val_test: datasplit to determine what needs to be loaded.

    :return: dataset_handler: dataset handler class instance
    """

    if dataset_config['dataset'] == 'kitti':
        dataset_handler = KittiDatasetHandler(dataset_config, train_val_test)
    elif dataset_config['dataset'] == 'bdd':
        dataset_handler = BddDatasetHandler(
            dataset_config, train_val_test)
    else:
        raise ValueError(
            'Invalid dataset type {}'.format(
                dataset_config['dataset']))

    return dataset_handler
