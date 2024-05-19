import dash
from dash import Dash, callback, Input, Output, dcc, html, dash_table       # Input, Output classes for reactive programming paradigm
import dash_bootstrap_components as dbc


# Initialize the app - incorporate css
# folgende css ist eine Variante - wir nehmen aber erstmal Bootstrap
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True, use_pages=True,
           pages_folder='pages')
app.title = "Synkyndineo Analytics"

app.layout = html.Div([
    html.H1('Synkyndineo Analytics - Stock price analysis', style={'text-align': 'center'}),
    html.Div([
        html.Div(
            # an dieser Stelle loopen wir durch unsere page-registry ...wir könnten auch Zeile für Zeile die pages
            # schreiben
            dcc.Link(f"{page['name']}", href=page["relative_path"])     # - {page['path']} # not necessary to see it
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])

#print(dash.page_registry.values())

if __name__ == '__main__':
    app.run(port='8054', debug=True)
