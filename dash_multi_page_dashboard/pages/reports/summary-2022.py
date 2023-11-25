import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1('2022 Summary'),
    html.Div("This is our page's content."),
])