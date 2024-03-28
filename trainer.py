import glob

import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self, data_root, dataset:tf.data.Dataset):
        self.data_root = data_root
        self.dataset = dataset
        self.class_list = np.array(sorted([x.split('/')[-2] for x in glob.glob(f'{self.data_root}/data/*/*.png')]))
        self.train_ds = self.dataset
