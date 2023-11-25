import dash
from dash import html

import dash_bootstrap_components as dbc

from .side_bar import sidebar

dash.register_page(__name__, name="Portfolio Allocation")


def layout():
    return dbc.Row(
        [dbc.Col(sidebar(), width=2),
         dbc.Col(html.Div('Topics Home Page'), width=2),
         dbc.Col(html.Div('Backtesting'), width=2),
         dbc.Col(html.Div('ML/DeepLearning'), width=2),
         dbc.Col(html.Div('Moving Averages Crossover'), width=2),
         dbc.Col(html.Div('Portfolio Allocation'), width=2)],
    )