"""
Enable multi page navigation:
The app source-code files should be structured as recommended by the Dash guidelines (https://dash.plotly.com/urls)

    app.py : this file is very basic and will only define the app variable needed by Flask.
    I’ve simply followed the Plotly Dash guidelines to create this file.

    index.py : this is a sort of navigator file, helping the app managing the URLs of the different pages.
    Also this file is very standard: I recommend to follow the Plotly Dash guidelines for it,
    as we just need to customise the pathnames to have this working.

    layouts.py : all pages html layouts will be stored in this file. Given that some components
    (like the header or the navbar) have to be replicated on each page, I’ve created some functions to return them,
    avoiding many repetitions within the code

    callbacks.py : all callbacks (the functions behind the Dash components,
    that define the user interactions with the graphs) will be stored in this file

    The favicon comes from here: https://twemoji-cheatsheet.vercel.app/
    https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f34b.png


    Sourcefiles here: https://github.com/gabri-al/corporate-dashboard
"""

import pandas as pd
from dash import Dash, callback, Input, Output, dcc, html    # Input, Output classes for reactive programming paradigm
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine_data = pd.read_csv('data/winequality-red.csv', delimiter=',')
# print(wine_data)

df = wine_data.copy()
df.set_index(pd.Index(range(len(df))), inplace=True)
print(df)

# array of non normalized quality-steps:
# print(df['quality'].sort_values().unique())

# sklearn LabelEncoder can be used to normalize labels.
quality_label = LabelEncoder()
df['quality'] = quality_label.fit_transform(df['quality'])
# array of normalized quality-steps:
# print(df['quality'].sort_values().unique())
# print(type(df['quality'].sort_values().unique()))

# split into Features and Labels:
X = df.drop('quality', axis=1)
y = df['quality']
print(df.columns)

# First, instantiate the Dash app:

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Random Forest Analytics: Understand Your Wine Quality!"

app.layout = dbc.Container(
        [
        html.H1('SciKit-Learn with Dash', style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col([
                html.Div('Select Test Size:'),
                dcc.Input(value=.2, type='number', debounce=True, id='test-size', min=.1, max=.9, step=.1)
            ], width=3),
            dbc.Col([
                html.Div("Select RandomForest n_estimators:"),
                dcc.Input(value=150, type='number', debounce=True, id='nestimator-size', min=10, max=200, step=1)
            ], width=3),
            dbc.Col([
                html.Div("Accuracy Score:"),
                html.Div(id='placeholder', style={'color':'blue'}, children="...rechnet")
            ], width=3)
        ], className='mb-3'),



# Under the first row, the Dash AG grid displays the data from our wine quality dataset. This grid allows
# the user to play with the way data is displayed, by moving columns or increasing the numbers of
# records displayed at a time.


    dag.AgGrid(
        id='grid',
        rowData=df.to_dict("records"),
        columnDefs=[{'field': i} for i in df.columns],
        columnSize='sizeToFit',
        style={"height": "310px"},
        dashGridOptions={'pagination': True, 'paginationPageSize': 5},
    ),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=px.histogram(df, 'fixed acidity', histfunc='avg')),
        ], width=6),
        dbc.Col([
            dcc.Graph(figure=px.histogram(df, 'pH', histfunc='avg')),
        ], width=6),
    ])

        ]
    )

@callback(
    Output('placeholder', 'children'),
    Input('test-size', 'value'),
    Input('nestimator-size', 'value')
)
def update_testing(test_size_value, nestimator_value):
    """
    Whenever someone change the Inputs on our Web-App, the decorator starts the function: in this function we
    gather a realtime-machine-learning algo of a Random-Forest-Classifier

    :param test_size_value:
    :param nestimator_value:
    :return:
    """
    # Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=42)

    # Apply standard scaling:
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.fit_transform(X_test)

    # Random forest classifier:
    rfc = RandomForestClassifier(n_estimators=nestimator_value)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    score = accuracy_score(y_test, pred_rfc)

    return score


if __name__ == '__main__':
    app.run_server(debug=True)
