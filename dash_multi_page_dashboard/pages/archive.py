import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, dash_table

import pandas as pd

dash.register_page(__name__,
                   path='/archive',
                   redirect_from=["/archive-2022", "/archive-2021"],
                   order=6)

# All CSS-Bootstrap-grid-system classes can be assigned to the html.Div([]) elements, within their className property.

colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'white': 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)',
    'slate-grey': '#5c6671',
    'slate-grey-invertiert': '#a3998e',
    'slate-grey-darker': '#4b5259',
    'dusk-white': '#CFCFCF',
    'back-white': '#F7F7F7',
    'anthrazitgrau': '#373f43',
    'invertiert': '#c8c0bc'

}

layout = dbc.Container([
    dbc.Row([
        html.Div([
            html.H1(children='This is our Archive page', className="text-center fs-3", style={'color': colors['invertiert']}),
            html.Div("This is our Archive page content. If you change a page's path, it's best practice to define "
                         "a redirect so users that go to old links don't get a '404 – Page not found'. "
                         "You can set additional paths to direct to a page using the redirects parameter." 
                         "This takes a list of all paths that redirect to this page.")
        ])
    ]),
    dbc.Row([
        dbc.Col([html.Div(children='2 Spalten')], width=2),
        dbc.Col([html.Div(children='8 Spalten'),
            dash_table.DataTable(data=[{'Population': 4.5, 'City': 'montreal', 'Country': 'canada'}, {'Population': 8, 'City': 'boston', 'Country': 'america'}], page_size=12, style_table={'overflowX': 'auto'}),
            dbc.Form(children=[html.Br(), dbc.Input(style={'width': '50%'}), dbc.Input(style={'width': '50%'})])
                 ], width=8),
        dbc.Col([html.Div(children='2 Spalten')], width=2),
    ]),
], fluid=True)




