import pandas as pd
from dash import Dash, Input, Output, dcc, html    # Input, Output classes for reactive programming paradigm


def setting_the_flask_server_and_app():
    """

    https://realpython.com/python-dash/

    and

    https://plotly.com/blog/build-python-web-apps-for-scikit-learn-with-plotly-dash/

    and for multi page apps:

    https://towardsdatascience.com/create-a-professional-dasbhoard-with-dash-and-css-bootstrap-e1829e238fc5

    Params:
    name – The name Flask should use for your app. Even if you provide your own ``server``, ``name`` will be used to help
    find assets. Typically ``__name__`` (the magic global var, not a string) is the best value to use.
    Default ``'__main__'``, env: ``DASH_APP_NAME``
    server – Sets the Flask server for your app. There are three options: ``True`` (default): Dash will create a
    new server ``False``: The server will be added later via ``app.init_app(server)`` where ``server`` is a
    ``flask.Flask`` instance. ``flask.Flask``: use this pre-existing Flask server.
    assets_folder – a path, relative to the current working directory, for extra files to be used in the browser.
    Default ``'assets'``. All .js and .css files will be loaded immediately unless excluded by ``assets_ignore``,
    and other files such as images will be served if requested.

    Style your page by adding the  "className"-Argument connected to css-sections in style.css

    Next step: Add Interactivity to Your Dash Apps Using Callbacks

    How to Define Callbacks
    You’ve defined how the user will interact with your application. Now you need to make your application react to
    user interactions. For that, you’ll use callback functions.

    Dash’s callback functions are regular Python functions with an app.callback decorator. In Dash, when an input changes,
    a callback function is triggered. The function performs some predetermined operations, like filtering a dataset,
    and returns an output to the application. In essence, callbacks link inputs and outputs in your app.
    :return:
    """

    data = (
        pd.read_csv('avocado.csv')
        # remove query since we use the interactive dropdowns; query u can use for rapid prototyping
        # .query("type == 'organic' and region == 'Indianapolis' and year == 2017 ")
        .assign(Date=lambda data: pd.to_datetime(data['Date'], format='%Y-%m-%d'))
        .sort_values(by='Date')
    )
    regions = data["region"].sort_values().unique()
    avocado_types = data["type"].sort_values().unique()

    # next dataset:
    data_wine = (pd.read_csv('winequality-red.csv'))
    alcohol = data_wine["quality"].sort_values().unique()
    quality = data_wine["alcohol"].sort_values().unique()
    print(quality, alcohol)

    external_stylesheets = [
        {"href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
            "rel": "stylesheet", },
    ]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.title = "Avocado Analytics: Understand Your Avocados!"

    app.layout = html.Div(
        children=[
            html.Div(
                children=[
                    html.P(children="🥑", className="header-emoji"),
                    html.H1(
                        children="Avocado Analytics", className="header-title"
                    ),
                    html.P(
                        children=(
                            "Analyze the behavior of avocado prices and the number"
                            " of avocados sold in the US between 2015 and 2018"
                        ),
                        className="header-description",
                    ),
                ],
                className="header",
            ),

            # divisions for dropdowns:

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(children="Region", className="menu-title"),
                            dcc.Dropdown(
                                id="region-filter",
                                options=[
                                    {"label": region, "value": region}
                                    for region in regions
                                ],
                                value="Albany",
                                clearable=False,
                                className="dropdown",
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div(children="Type", className="menu-title"),
                            dcc.Dropdown(
                                id="type-filter",
                                options=[
                                    {
                                        "label": avocado_type.title(),
                                        "value": avocado_type,
                                    }
                                    for avocado_type in avocado_types
                                ],
                                value="organic",
                                clearable=False,
                                searchable=False,
                                className="dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children="Date Range", className="menu-title"
                            ),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=data["Date"].min().date(),
                                max_date_allowed=data["Date"].max().date(),
                                start_date=data["Date"].min().date(),
                                end_date=data["Date"].max().date(),
                            ),
                        ]
                    ),
                ],
                className="menu",
            ),


            # divisions for graphs:

            html.Div(
                children=[
                    html.Div(
                        children=dcc.Graph(
                            id="price-chart",
                            config={"displayModeBar": False},
                            figure={
                                "data": [
                                    {
                                        "x": data["Date"],
                                        "y": data["AveragePrice"],
                                        "type": "lines",
                                        "hovertemplate": (
                                            "$%{y:.2f}<extra></extra>"
                                        ),
                                    },
                                ],
                                "layout": {
                                    "title": {
                                        "text": "Average Price of Avocados",
                                        "x": 0.05,
                                        "xanchor": "left",
                                    },
                                    "xaxis": {"fixedrange": True},
                                    "yaxis": {
                                        "tickprefix": "$",
                                        "fixedrange": True,
                                    },
                                    "colorway": ["#17b897"],
                                },
                            },
                        ),
                        className="card",
                    ),
                    html.Div(
                        children=dcc.Graph(
                            id="volume-chart",
                            config={"displayModeBar": False},
                            figure={
                                "data": [
                                    {
                                        "x": data["Date"],
                                        "y": data["Total Volume"],
                                        "type": "lines",
                                    },
                                ],
                                "layout": {
                                    "title": {
                                        "text": "Avocados Sold",
                                        "x": 0.05,
                                        "xanchor": "left",
                                    },
                                    "xaxis": {"fixedrange": True},
                                    "yaxis": {"fixedrange": True},
                                    "colorway": ["#E12D39"],
                                },
                            },
                        ),
                        className="card",
                    ),
                ],
                className="wrapper",
            ),
        ]
    )

    @app.callback(
        Output("price-chart", "figure"),
        Output("volume-chart", "figure"), # see above the 'id', figure ist the parameter figure above
        Input("region-filter", "value"),
        Input("type-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    )
    def update_charts(region, avocado_type, start_date, end_date):
        filtered_data = data.query(
            "region == @region and type == @avocado_type"
            " and Date >= @start_date and Date <= @end_date"
        )
        price_chart_figure = {
            "data": [
                {
                    "x": filtered_data["Date"],
                    "y": filtered_data["AveragePrice"],
                    "type": "lines",
                    "hovertemplate": "$%{y:.2f}<extra></extra>",
                },
            ],
            "layout": {
                "title": {
                    "text": "Average Price of Avocados",
                    "x": 0.05,
                    "xanchor": "left",
                },
                "xaxis": {"fixedrange": True},
                "yaxis": {"tickprefix": "$", "fixedrange": True},
                "colorway": ["#17B897"],
            },
        }

        volume_chart_figure = {
            "data": [
                {
                    "x": filtered_data["Date"],
                    "y": filtered_data["Total Volume"],
                    "type": "lines",
                },
            ],
            "layout": {
                "title": {"text": "Avocados Sold", "x": 0.05, "xanchor": "left"},
                "xaxis": {"fixedrange": True},
                "yaxis": {"fixedrange": True},
                "colorway": ["#E12D39"],
            },
        }
        return price_chart_figure, volume_chart_figure
    return app


if __name__ == '__main__':
    app = setting_the_flask_server_and_app()
    app.run_server(debug=True)

