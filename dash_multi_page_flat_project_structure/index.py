"""
index.py : this is a sort of navigator file, helping the app managing the URLs of the different pages.
Also, this file is very standard: I recommend to follow the Plotly Dash guidelines for it,
as we just need to customise the pathnames to have this working.

Enable multi-page navigation:

(https://dash.plotly.com/urls)
"""

import dash
from dash import dcc, html, callback, Input, Output, dash_table

from dash_multi_page_flat_project_structure.app import app

from layouts import sales, page2, page3
import dash_multi_page_flat_project_structure.callbacks

app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname')) # 1. 'id' in layout, figure ist the parameter figure in layouts
def display_page(pathname):
    if pathname == '/apps/sales-overview':
        return sales
    elif pathname == '/apps/page2':
        return page2
    elif pathname == '/apps/page3':
        return page3
    else:
        return sales    # This is the home page


if __name__ == '__main__':
    app.run(port='8050', debug=True)