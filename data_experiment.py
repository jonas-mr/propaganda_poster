#Essentials
import glob
import pathlib
import pickle

import numpy as np
import pandas as pd
#Clustering
from sklearn.cluster import KMeans, DBSCAN
from subclu import py_subclu
#Logging Module
import logging



class DataExperiment:
    def __init__(self, data_root, model='vgg16'):
        self.data_root = pathlib.Path(data_root).with_suffix('')
        self.file_list = np.array(sorted(['/'.join(x.split('/')[-2:]) for x in glob.glob(f'{self.data_root}/data/*/*.png')]))
        self.reduced_features = self.load_results(model)


    def load_results(self, model):
        try:
            with open(f'{self.data_root}/results/exp_result_{model}.pkl', mode='rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            filelist = glob.glob(f'{self.data_root}/results/exp_result_*.pkl')
            logging.warning(f'Could not find result pickle with name "exp_result_{model}.pkl". Loading first result pickle that is found which is: "{filelist[0]}".')
            with open(filelist[0], mode='rb') as f:
                return pickle.load(f)

    def k_means(self, k):

        logging.info('initiating K-Means clustering')
        kmodel = KMeans(init = 'k-means++', n_clusters = k)
        kmodel.fit(self.reduced_features )

        return kmodel

    def dbscan(self,eps, min_samples=2):
        '''
        Calls the sklearn DBSCAN Algorithm
        :param eps: maximal Distance for two points to belong to the same cluster
        :return:
        '''
        logging.info('initiating DBSCAN clustering')
        dbmodel = DBSCAN(eps=eps, min_samples=min_samples)
        dbmodel.fit(self.reduced_features)
        logging.info(f'completed DBSCAN clustering for eps={eps}')

        return dbmodel


    def subclu(self, eps, min_samples=2):
        '''
        An Implementation of the subclu Algorithm for high dimensional data
        :param DB:
        :param Eps:
        :param MinPts:
        :return:
        '''
        logging.info('initiating SUBCLU clustering')
        feature_df = pd.DataFrame(self.reduced_features)
        C,S,df_C = py_subclu.fullSUBCLU(feature_df, eps, m=min_samples, draw_results=False)
        return feature_df, C,S,df_C