import dash
from dash import html

dash.register_page(__name__)

layout = html.H1("Sorry mate - This is our custom 404 content - kindly try another path\n"
                 "Could it be that you commented-out the registry of your demanded page???")