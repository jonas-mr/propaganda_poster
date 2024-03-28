import glob
import json
import os

import numpy as np
from dash import Dash, dcc, html, Input, Output, no_update, callback, ctx
from PIL import Image
import io
import base64
from visualizer import Visualizer

data_root = './datasets/propagandaSet'
model_list = [x.split('/')[-1].rstrip('.pkl').split('_', maxsplit=2)[-1] for x in glob.glob(f'{data_root}/results/*.pkl')]
file_list = np.array(sorted(['/'.join(x.split('/')[-2:]) for x in glob.glob(f'{data_root}/data/*/*.png')]))
app = Dash(__name__, external_stylesheets=['/assets/styles.css'])

app.layout = html.Div(
    children=[
        # Header
        html.Div([
            html.H1("Propaganda Cluster"),
            html.Hr()
        ], style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label('Choose the model and the number of features to be used.'),
            dcc.Dropdown(id="model-dropdown", options=model_list, value='org'),
        ], style={'width': '50%', 'float': 'left'}),
        # Options
        html.Div([
            html.Div([
                html.Label('Choose the algorithm to cluster the dataset.'),
                dcc.Dropdown(id="algorithm-dropdown", options=['k Means', 'DBSCAN', 'SUBCLU'], value='SUBCLU'),
            ], style={'width': '50%', 'float': 'left'}),
            html.Div([
                html.Label('Choose the number of clusters for the k Means algorithm'),
                dcc.Slider(min=2, max=12, marks={i: str(i) for i in range(13)}, step=1, value=2,
                           id='num-clusters-slider')
            ],
                id='k-Means-slider', style={'width': '50%', 'float': 'right'}
            ),
            html.Div([
                html.Label('Choose the maximum distance for the DBSCAN algorithm'),
                dcc.Input(type='number', min=0, step=1,value=5, id='max-distance'),
                html.Label('Choose the minimal number of neighbors for the DBSCAN algorithm'),
                dcc.Input(type='number', min=1, step=1,value=2, id='min-pts'),
                html.Button('Los!', id='max-distance-button', n_clicks=0)
            ],
                id='dbscan-selector', style={'width': '45%', 'float': 'right'}
            ),
        ], style={'marginBottom': '30px'}),

        # Main Content
        html.Div([
            #Left section
            html.Div([
                dcc.Graph(id="propaganda_graph",  clear_on_unhover=True),
                dcc.Markdown('Made by MRJonas', id='stats')
            ], style={'width': '50%', 'float': 'left', 'padding': '10px', 'boxSizing': 'border-box'},
                id='left-section'),
            #Right section
            html.Div([
                #Image
                html.Div([
                    html.Img(id='hover-data', src='', style={'width': 'content-width'})
                ],style={'text-align': 'center'}),

                dcc.Markdown('',id='image_metadata', style={'overflow':'scroll'})
            ], style={'width': '50%', 'float': 'left', 'padding': '10px', 'boxSizing': 'border-box'},
                id='right-section')
        ],id='main-content', style={'margin': '20px 0'}),


    ]
)

@callback(
    Output("propaganda_graph", "figure"),
    Output('stats', 'children'),
    Input('algorithm-dropdown', 'value'),
    Input('num-clusters-slider', 'value'),
    Input('max-distance', 'value'),
    Input('min-pts', 'value'),
    Input('max-distance-button', 'n_clicks'),
    Input('model-dropdown', 'value')
)
def update_figure(algorithm, numClusters, maxDistance, min_pts, mdb, model):
    vis = Visualizer(data_root, model)
    if algorithm == 'k Means':
        cluster_model = vis.experiment_result.k_means(numClusters)
        title = f'### k Means Cluster with k={numClusters}.\n'
        numClusters = cluster_model.labels_.max() + 1
    elif algorithm in ('DBSCAN', 'SUBCLU'):
        if ctx.triggered_id in ('max-distance-button', 'algorithm-dropdown'):
            if type(maxDistance) is int:
                maxDistance = float(maxDistance)
            if algorithm == 'DBSCAN':
                cluster_model = vis.experiment_result.dbscan(maxDistance, min_pts)
                numClusters = cluster_model.labels_.max() + 2
            elif algorithm == 'SUBCLU':
                cluster_model = vis.experiment_result.subclu(maxDistance, min_pts)
                title = f'### SUBCLU Algorithm with maxDistance={maxDistance} and minPts={min_pts}.\n'
                return vis.visualize_subclu(cluster_model), title
            title = f'### DBSCAN Algorithm with maxDistance= {maxDistance} and minPts={min_pts}.\n'
        else:
            return no_update
    description = title + f'#### Number of Clusters: {numClusters}\n'
    for cluster in range(-1, numClusters):
        if algorithm == 'DBSCAN' and cluster == -1:
            description += f'#### {np.count_nonzero(cluster_model.labels_ == cluster)} Images are marked as noise\n'
        elif cluster > -1:
            description += f'#### {np.count_nonzero(cluster_model.labels_ == cluster)} Images in Cluster {cluster + 1}\n'
        else:
            continue
    return vis.reduce_visual_features(cluster_model),description+'Made by MRJonas'

@callback(
    Output("hover-data", "src"),
    Output("image_metadata", "children"),
    Input("propaganda_graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    num = hover_data["pointNumber"]



    im_file = file_list[num]
    im_matrix = np.array(Image.open(str(f'{data_root}/data/{im_file}')))
    try:
        with open (str(f'{data_root}/data/{im_file}'.rstrip(".png")+'.json'), 'r') as file:
            metadata = json.load(file)
        metadata = render_metadata(metadata)
    except FileNotFoundError:
        metadata = ''

    im_url = np_image_to_base64(im_matrix)
    return im_url, metadata

@callback(
    Output("k-Means-slider", "style"),
    Output("dbscan-selector", "style"),
    Input('algorithm-dropdown', 'value')
)
def show_hide_element(chosen_algorithm):
    if chosen_algorithm in ('DBSCAN', 'SUBCLU'):
        kmeans_style = {'display': 'none','width': '50%', 'float': 'right', 'rightPadding':'5%'}
        dbscan_style = {'display': 'inline','width': '50%', 'float': 'right', 'rightPadding':'5%'}
        return kmeans_style, dbscan_style
    elif chosen_algorithm == 'k Means':
        dbscan_style = {'display': 'none','width': '50%', 'float': 'right', 'rightPadding':'5%'}
        kmeans_style = {'display': 'inline','width': '50%', 'float': 'right', 'rightPadding':'5%'}
        return kmeans_style, dbscan_style

def render_metadata(metadata):
    """

    :param metadata:
    :return:
    """

    meta_string =  f"# Metadata\n"
    for md in metadata.keys():
        meta_string = meta_string + f"### {md}\n{metadata[md]}\n"

    return meta_string


def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

app.run(debug=True)