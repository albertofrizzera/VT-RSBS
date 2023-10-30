# -------------------------------------------------------------------------
# Author:   Alberto Frizzera
# Date:     02/09/2023, info@albertofrizzera.com
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import pandas as pd
import numpy as np
import pickle
import random
import time

import dash
from dash import Dash, html, dcc, dash_table, callback, MATCH, ALL, ctx
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import cv2
import imutils
import h5py

from plot.web_app.app_utils import get_figure

dash.register_page(
    __name__,
    path='/',
    title='SatCLIP',
    name='SatCLIP'
)
from dataset.dataset import *

# Initialization global variables
dataset = globals()["SatCLIP"](None, "train", "val", "test", label_type="sentence")
blank_fig = go.Figure()
blank_fig.update_xaxes(visible=False)
blank_fig.update_yaxes(visible=False)
blank_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# Output
title = dbc.Row(html.H1("Dataset Viewer"),className="text-center mt-4 mb-1")
subtitle = dbc.Row(dbc.Col(html.H4("SatCLIP")),
                   style={'color': 'gray'},
                   className="text-center mt-1 mb-5")
selection_box = dbc.Row([dbc.Col([html.Br(),html.Br(),
                                html.H6('Dataset:'),
                                dcc.RadioItems(labelStyle={'display': 'block', "color": "black"},
                                                options=["BigEarthNet","EuroSAT","MLRSNet","NWPU","OPTIMAL_31","PatternNet","RESISC45","RSI_CB256","RSICD","RSITMD","SIDNEY","SIRI_WHU","UCM","WHU_RS19"],
                                                className="text-center mb-4, mt-4", 
                                                inputStyle={"margin-right": "20px"},
                                                value="SatCLIP",
                                                id="radio_items"),
                                ]),
                         dbc.Col([html.Br(),html.Br(),
                                html.H6('Label (Optional):'),
                                html.Br(),
                                dcc.Input(value="", id="input_label", placeholder="Enter label:"),
                                html.Br(),html.Br(),
                                dbc.Button("Select random sample", color="primary", id="random_image"),
                                ]),
                        #  dbc.Col([html.Br(),html.Br(),
                        #         html.H6('Dataset index:'),
                        #         html.Br(),
                        #         dcc.Input(id="input_index", placeholder="Enter index:"),
                        #         html.Br(),html.Br(),
                        #         dbc.Button("Select custom sample", color="primary", id="custom_image"),
                        #         ]),
                        dbc.Row([html.Br(),html.Br(),
                                html.H6("Available labels: "),
                                html.H6(str(dataset.unique_labels), id="unique_labels", style={'color': 'black'}),
                                html.Br(),html.Br(),html.Br(),
                                html.H6("Selected label: "),
                                html.H6("", style={'color': 'black'}, id="label_text"),
                                html.H6("", style={'color': 'red'}, id="error"),
                                dbc.Col(dcc.Graph(figure=blank_fig, id='graph_image')),
                                ], style={"margin-top": "20px"}),
                        ],
                        className="text-center",
                        style={'color': 'gray'},)
layout = dbc.Container(fluid=True, 
                       children=[title,
                                 subtitle,
                                 selection_box,
                                ])


@callback(Output('unique_labels', 'children'),
          Input('radio_items', 'value'),
          prevent_initial_call=True)
def select_dataset(dataset_name):
    global dataset
    if dataset_name=="SatCLIP":
        dataset = globals()[dataset_name](None, "train", "val", "test", label_type="sentence")
    else:
        dataset = globals()[dataset_name](None, "train", "val", "test")
    try:
        unique_labels = str(dataset.unique_labels)
    except:
        unique_labels = "Too long sentences..."
    return unique_labels


@callback([Output('graph_image', 'figure'),
           Output("label_text", "children"),
           Output("error", "children"),
           ],
          [Input('input_label', 'value'),
           Input('random_image', 'n_clicks'),
          ], prevent_initial_call=True)
def plot_image(label, random_image):
    global dataset
    triggered_id = ctx.triggered_id
    if triggered_id == 'random_image':
        if label!="":
            try:
                image_tensor, label_text = dataset.custom_label(label)
            except:
                return [blank_fig, "", "Sample not found!"]
        else:
            image_tensor, label_text = dataset[random.randint(0, len(dataset)-1)]
            image_tensor = image_tensor["image"]
        
        image_array = np.moveaxis(image_tensor.numpy(), 0, -1)
        fig_image = get_figure(image_array)
        return fig_image, label_text, ""
    else:
        return [dash.no_update]*3