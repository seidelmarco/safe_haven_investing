from dash import Dash, html, dcc, callback, Input, Output

import dash_bootstrap_components as dbc


"""
This file is very basic and will only define the app variable needed by Flask.

Im Tutorial gibt es die offizielle Bootstrap .css in assets - wenn ich diese nutze, brauche ich hier nicht
auf external_stylesheets=[dbc.themes.BOOTSTRAP] verlinken
"""
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Synkyndineo Analytics"
