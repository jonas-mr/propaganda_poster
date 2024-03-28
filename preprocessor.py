import glob
import logging
import pathlib
import pickle

import numpy as np
import keras.utils
import tensorflow as tf
#Dimension Reduction
from sklearn.decomposition import PCA
#Models
from keras.applications.vgg16 import VGG16
from keras.models import Model


class Preprocessor:
    def __init__(self, data_root, dataset, model, components):
        self.data_root = pathlib.Path(data_root).with_suffix('')
        self.file_list = np.array(sorted(['/'.join(x.split('/')[-2:]) for x in glob.glob(f'{self.data_root}/data/*/*.png')]))
        self.dataset = dataset
        self.model = model
        self.preprocessed_dataset = self.prepare_dataset_for_model(model)
        self.features = self.feature_extraction()
        self.reduced_features = self.feature_reduction(components)
        self.save_results(model, self.reduced_features)
    def prepare_dataset_for_model(self, model):
        preprocessed_dataset = self.dataset.map(lambda x, y, z: (keras.applications.vgg16.preprocess_input(x), y, z),
                                                num_parallel_calls=tf.data.AUTOTUNE)

        logging.info(f'Preparing dataset for transfer learning on {model.rstrip("_")}')
        #logging.info(f'Saving dataset to {self.data_root}/dataset/{model}')
        #preprocessed_dataset.save(f'{self.data_root}/dataset/{model}')
        image_batch, file_batch, label_batch = next(iter(preprocessed_dataset))
        logging.info(f'Shape of image batch is: {image_batch.shape}. Shape of label batch is: {file_batch.shape}')

        return preprocessed_dataset


    def feature_extraction(self):
        """
        extract features from preprocessed dataset using the specified model
        :param components: number of dimensions to
        :return:
        """
        logging.info('Beginning feature extraction')

        logging.info('loading model for transfer learning')
        if self.model == 'vgg16':
            model = VGG16()
        else:
            model = VGG16()
            #ToDo: add additional models for transfer learning

        model = Model(inputs=model.input, outputs=model.layers[-2].output)
        features = model.predict(self.preprocessed_dataset, batch_size=32, use_multiprocessing=True)
        self.save_results(f'{self.model}', features)
        return features


    def feature_reduction(self, components=100):
        """
        Principal component reduction using PCA.
        :param components: number of dimensions to reduce the feature vector to
        :return:
        """
        pca_2 = PCA(n_components=components)
        reduced_features = pca_2.fit_transform(self.features)
        reduced_features = np.ascontiguousarray(reduced_features)
        logging.info(f'Components before PCA:{self.features.shape[1]}')
        logging.info(f'Components after PCA:{pca_2.n_components}')
        logging.info(f'Explained variation per principal component: {pca_2.explained_variance_ratio_}')
        logging.info(f'Cumulative variance explained by {components} principal components: {np.sum(pca_2.explained_variance_ratio_):.2%}')
        return reduced_features
    def save_results(self, model, features):
        """
        saves extracted features to a pickle file
        :param model: the model used for predicting features
        :param features: the feature vector to save
        :return: None
        """
        with open(f'{self.data_root}/results/exp_result_{model}.pkl', 'wb') as f:
            pickle.dump(features, f)

