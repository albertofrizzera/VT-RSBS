# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     02/09/2023
# -------------------------------------------------------------------------

import os
import sys
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
import numpy as np
import time
import gc
import pandas as pd
import pickle
from dotenv import load_dotenv
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def get_figure(image_array):
    fig = go.Figure()
    fig.add_trace(go.Image(z=image_array))
    fig.update_xaxes(range=[0,image_array.shape[1]])
    fig.update_yaxes(range=[image_array.shape[0],0])
    fig.update_layout(
                # title={'text': 'SatCLIP',
                #         'xanchor': 'center',
                #         'yanchor': 'top',
                #         #    'y':0.9,
                #         'x':0.5,},
                # width=1000,
                height=600,
                # legend_xanchor="right",
                # legend_yanchor="top",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(r=20, l=20, b=20, t=20))
    return fig