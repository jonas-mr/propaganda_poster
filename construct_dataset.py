import glob

import tensorflow as tf
import pathlib
import os
import numpy as np
import logging

class ConstructDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.file_list = np.array(sorted([x.split('/')[-1] for x in glob.glob(f'{self.data_root}/data/*/*.png')]))
        self.class_list = np.array(sorted([x.split('/')[-2] for x in glob.glob(f'{self.data_root}/data/*/*.png')]))
        self.dataset = self.load_or_create_dataset()

    def load_or_create_dataset(self, model='org'):
        """
        loads the specified dataset
        :param model:
        :return:
        """

        try:
            dataset = tf.data.Dataset.load(f'{self.data_root}/dataset/{model}')
            #image_batch, file_batch, label_batch = next(iter(dataset))
            #logging.info(f'Shape of image batch is: {image_batch.shape}. Shape of label batch is: {file_batch.shape}')
            return dataset
        except tf.errors.NotFoundError:
            logging.warning(f'No suitable Dataset found for given [modelname_components]: {model}. Trying to generate Dataset from directory: {self.data_root}/data/')

            data_dir = pathlib.Path(f'{self.data_root}/data/').with_suffix('')
            image_list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.png'), shuffle=False)
            image_list_ds = image_list_ds.shuffle(len(image_list_ds), reshuffle_each_iteration=False)

            logging.info(f'Found {len(list(image_list_ds))} images in {data_dir}')
            batch_size = 32
            image_size = [224,224]

            def get_label(file_path):
                parts = tf.strings.split(file_path, os.path.sep)
                parts = tf.strings.reduce_join(parts[-2], separator=os.path.sep)
                one_hot = parts == self.class_list
                return tf.argmax(one_hot)

            def get_file_name(file_path):
                parts = tf.strings.split(file_path, os.path.sep)
                parts = tf.strings.reduce_join(parts[-1], separator=os.path.sep)
                one_hot = parts == self.file_list
                return tf.argmax(one_hot)

            def decode_img(image):
                img = tf.io.decode_png(image, channels=3)
                return tf.image.resize(img, image_size)

            def process_path(file_path):
                label = get_label(file_path)
                file_name = get_file_name(file_path)
                img = tf.io.read_file(file_path)
                img = decode_img(img)
                return img, file_name, label

            dataset = image_list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            logging.info(f'Creation of dataset finished. Saving to {self.data_root}/datasets/{model}')
            dataset.save(f'{self.data_root}/dataset/org')
            image_batch, file_batch, label_batch = next(iter(dataset))
            logging.info(f'Shape of image batch is: {image_batch.shape}. Shape of label batch is: {file_batch.shape}')
            return dataset

