# -------------------------------------------------------------------------
# Author:   Alberto Frizzera, info@albertofrizzera.com
# Date:     02/09/2023
# -------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, ctx
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_enterprise_auth as auth
import plotly.graph_objects as go
import dash_auth
import flask
from dotenv import load_dotenv
import webbrowser

load_dotenv()

# Check port availability -> sudo lsof -i :8080
# Close port -> kill PID
theme = dbc.themes.BOOTSTRAP
css = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'

# App Instance
app = dash.Dash(name="SatCLIP",
                external_stylesheets=[theme, css],
                use_pages=True,
                pages_folder=os.path.join(os.path.dirname(__file__), "pages"))
app.config['suppress_callback_exceptions']=True

# VALID_USERNAME_PASSWORD_PAIRS = {
#     'alberto': 'albertodl',
#     'guest': 'guestdl'
# }
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

app.layout = dbc.Container(fluid=True, children=[
	dash.page_container
])

if __name__ == "__main__":
    # webbrowser.open_new('http://127.0.0.1:8050/')
    # app.run_server(debug=True, port=80, host="0.0.0.0")
    app.run_server(debug=True, port=8080)