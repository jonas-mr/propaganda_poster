import pickle

import numpy as np
import plotly.graph_objects as go
import umap
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from data_experiment import DataExperiment


class Visualizer:
    def __init__(self, data_root, model):
        self.data_root = data_root
        self.experiment_result = DataExperiment(data_root, model)


    def load_results(self, model):
        with open(f'{self.data_root}/results/exp_result_{model}.pkl', mode='rb') as f:
            return pickle.load(f)

    def visualizing_results(self, cluster_model) -> go.Figure:
        """ Visualizing the clusters
        """
        x = self.experiment_result.reduced_features[:, 0]
        y = self.experiment_result.reduced_features[:, 1]
        z = self.experiment_result.reduced_features[:, 2]

        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=2, color=cluster_model.labels_))])

        return fig

    def reduce_visual_features(self, cluster_model, components=3) -> go.Figure:
        """
        Visualizing the clusters using PCA dimensionality reduction after applying the algorithm.
        :param cluster_model:
        :param components:
        :return:
        """
        reduced_features = PCA(components).fit_transform(self.experiment_result.reduced_features)
        x = reduced_features[:, 0]
        y = reduced_features[:, 1]
        z = reduced_features[:, 2]

        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=2, color=cluster_model.labels_))])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        return fig

    def visualize_subclu(self, cluster_model, components=3) -> go.Figure:

        X, C,S,df_C = cluster_model
        return self.draw_subspaces_clusters_plotly(X, X, df_C)


    def draw_subspaces_clusters_plotly(self, X, X_scaled, df_C, limit=None):
        """
        A modified version of the py_subclu.draw_subspaces_clusters function by Emmanuel Doumard.
        Draw a figure for each subspace, coloring each dataset
        For subspaces of dimension 1, using jitter to get an idea of the density
        For subspaces of dimension 2, use directly the 2 dimensions
        For subspaces of dimension 3, plot both the 3D representation and the UMAP projection
        For subspaces of dimension >3, plot only the UMAP projection
        """

        subspaces_ordered_list = df_C.sort_values("Subspace quality", ascending=False)["Subspace"].drop_duplicates()

        if not limit:
            limit = subspaces_ordered_list.shape[0]

        for s in subspaces_ordered_list[:limit]:
            if len(s) == 1:  # 1-Dim subspaces
                rand_jitter = (np.random.rand(X.shape[0]) - 0.5) * 0.2
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X.loc[:, list(s)[0]], y=rand_jitter, mode='markers'))
                for ic in df_C.loc[df_C["Subspace"] == s, "Indices"]:
                    fig.add_trace(go.Scatter(x=X.loc[ic, list(s)[0]], y=rand_jitter[ic], mode='markers'))
                fig.update_layout(title="Subspace " + str(s) + " with quality " + str(
                    df_C.loc[df_C["Subspace"] == s, "Subspace quality"].mean()),
                                  xaxis=dict(title=list(s)[0]),
                                  yaxis=dict(title="Jitter"))
                return fig

            elif len(s) == 2:  # 2-Dim subspaces
                labels = list(s)
                x_label = labels[0]
                y_label = labels[1]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X.loc[:, x_label], y=X.loc[:, y_label], mode='markers'))
                for ic in df_C.loc[df_C["Subspace"] == s, "Indices"]:
                    fig.add_trace(go.Scatter(x=X.loc[ic, x_label], y=X.loc[ic, y_label], mode='markers'))
                fig.update_layout(title="Subspace " + str(s) + " with quality " + str(
                    df_C.loc[df_C["Subspace"] == s, "Subspace quality"].mean()),
                                  xaxis=dict(title=x_label),
                                  yaxis=dict(title=y_label))
                return fig

            elif len(s) == 3:  # 3-dim subspaces
                labels = list(s)
                x_label = labels[0]
                y_label = labels[1]
                z_label = labels[2]

                total_ic = []
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=X.loc[:, x_label], y=X.loc[:, y_label], z=X.loc[:, z_label], mode='markers'))
                for i, ic in enumerate(df_C.loc[df_C["Subspace"] == s, "Indices"]):
                    fig.add_trace(
                        go.Scatter3d(x=X.loc[ic, x_label], y=X.loc[ic, y_label], z=X.loc[ic, z_label], mode='markers',
                                     marker=dict(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][
                                         (i + 1) % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])])))
                    total_ic.append(ic)
                nc = np.setdiff1d(X.index, np.array([ic for sublist in total_ic for ic in sublist]))
                fig.add_trace(
                    go.Scatter3d(x=X.loc[nc, x_label], y=X.loc[nc, y_label], z=X.loc[nc, z_label], mode='markers'))

                reducer = umap.UMAP().fit(X_scaled.loc[:, s])
                fig.add_trace(go.Scatter(x=reducer.embedding_[:, 0], y=reducer.embedding_[:, 1], mode='markers'))
                for ic in df_C.loc[df_C["Subspace"] == s, "Indices"]:
                    fig.add_trace(go.Scatter(x=reducer.embedding_[ic, 0], y=reducer.embedding_[ic, 1], mode='markers'))
                fig.update_layout(title="Subspace " + str(s) + " with quality " + str(
                    df_C.loc[df_C["Subspace"] == s, "Subspace quality"].mean()))
                return fig

            else:  # >3-dim subspaces
                reducer = umap.UMAP().fit(X_scaled.loc[:, s])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=reducer.embedding_[:, 0], y=reducer.embedding_[:, 1], mode='markers'))
                for ic in df_C.loc[df_C["Subspace"] == s, "Indices"]:
                    fig.add_trace(go.Scatter(x=reducer.embedding_[ic, 0], y=reducer.embedding_[ic, 1], mode='markers'))
                fig.update_layout(title="Subspace " + str(s) + " with quality " + str(
                    df_C.loc[df_C["Subspace"] == s, "Subspace quality"].mean()),
                                  xaxis=dict(title="UMAP X"),
                                  yaxis=dict(title="UMAP Y"))
                return fig
