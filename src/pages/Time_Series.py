import numpy as np
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go

import dash
from dash import Dash, html, dcc, callback, Output, Input

dash.register_page(__name__)

layout = [
    html.Div(
        children = [
            
            html.H1(children = 'Applied Econometric Models', style = {'textAlign' : 'center', "text-decoration": "underline"}),
            html.Div(dcc.Markdown(r'$Y_{t} = \sum_{i=1}^{p} \phi_{i} Y_{t-i} + \epsilon_{t}$', mathjax = True), style = {'textAlign' : 'center', "font-size" : "x-large"}),

            html.Div(children = [html.Label("Select Econometric Model: "), dcc.Dropdown(["Autoregressive Model", "Moving Average Model", "Error Correction Model"], value = 'Autoregressive Model', id = 'dropdown-selection')], style = {'margin':'auto', 'width': '20%'}),
            
            html.Div(dcc.Graph(id = 'autoregressive-figure'), style = {'margin':'auto', 'width': '75%'}),
            html.Div(dcc.Graph(id = 'residuals-figure'), style = {'margin':'auto', 'width': '75%'}),

        ]
    )
]

@callback([Output('autoregressive-figure', 'figure'), Output('residuals-figure', 'figure') ], Input('dropdown-selection', 'value'))
def plot_data(value):
    """
    
    """

    np.random.seed(1)
    ar_process = np.zeros(1000)
    ar_process[0] = 25.0

    for i in range(1, 1000):
        ar_process[i] = 0.2 + ar_process[i - 1] + np.random.normal(0, 1) 

    X = np.arange(0, 1000, 1).reshape(-1, 1)
    reg = LinearRegression().fit(X, ar_process)
    pred = reg.predict(X)

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x = np.arange(0, 1000, 1),
            y = ar_process,
            name = "MEV", 
            marker_color = "blue"
        )
    )

    fig1.add_trace(
        go.Scatter(
            x = np.arange(0, 1000, 1),
            y = pred,
            name = "Linear Model",
            marker_color = "red"
        )
    )

    fig1.update_layout(title = "Autoregressive Process", xaxis_title = "Date", yaxis_title = "Macro Economic Variable", template = 'seaborn', paper_bgcolor = 'rgba(0, 0, 0, 0)')

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x = np.arange(0, 1000, 1),
            y = pred - ar_process,
            marker_color = "blue"
        )
    )

    fig2.update_layout(title = "Autoregressive Model Residuals", xaxis_title = "Date", yaxis_title = "Residual", template = 'seaborn', paper_bgcolor = 'rgba(0, 0, 0, 0)')

    return fig1, fig2



