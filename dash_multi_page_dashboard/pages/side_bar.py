import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def sidebar():
    """
    To access dash.page_registry from within a file in the pages directory, you'll need to use it within a function.

    Here in this case page_registry is not outside but inside the layout-function

    We see then a nice linkage in the web-app
    :return:
    """
    return html.Div(
        html.Nav([
            dbc.NavLink(
                html.Div(page['name'], className='ms-2'),
                href=page['path'],
                active='exact',
            )
            for page in dash.page_registry.values()
            if page['path'].startswith('/topic')
        ],
            className='bg-light',
        )
    )