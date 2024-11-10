import numpy as np
import pandas as pd
from functools import reduce

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, callback, Output, Input
from dash import dash_table

from sklearn.linear_model import LinearRegression

app = Dash()
server = app.server
external_stylesheets = ['mobile.css']

###################################################################################################
#
###################################################################################################
train = pd.read_csv("Data/TrainSet.csv").dropna()
test = pd.read_csv("Data/TestSet.csv")
target = pd.read_csv("Data/NetInterestIncome.csv")

test["NII"] = [None] * len(test)

target['DATE'] = pd.to_datetime(target['DATE'])

target['quarter'] = target['DATE'].dt.quarter
target['year'] = target['DATE'].dt.year

target['period_quarter'] = target['year'].astype(str) + ' Q' + target['quarter'].astype(str)
target = target.drop(["DATE", "quarter", "year"], axis = 1)
target.columns = ["NII", "Date"]

merged = pd.merge(train, target, how = "inner", on = "Date")
merged = pd.concat([merged, test])
###################################################################################################
#
###################################################################################################

app.layout = [
    html.Div(children = [

        html.H1(children = 'Comprehensive Capital Analysis and Review', style = {'textAlign' : 'center', "text-decoration": "underline"}),

        html.Div(children = [html.Label("Select Macro Economic Variable: "), dcc.Dropdown(train.columns[2:], value = 'House Price Index (Level)', id = 'dropdown-selection')], style = {'margin':'auto', 'width': '20%'}),
        html.Div(dcc.Graph(id = 'mev-figure'), style = {'margin':'auto', 'width': '75%'}),

        html.Div(dcc.Graph(id = 'autoregressive-model'), style = {'margin':'auto', 'width': '75%'}),

        # html.Div(dash_table.DataTable(merged.to_dict('records'), [{"name": i, "id": i} for i in merged.columns])),

    ], style = {'background-color' : 'whitesmoke'})
]

@callback(Output('mev-figure', 'figure'), Input('dropdown-selection', 'value'))
def plot_data(col):
    """
    
    """

    train = merged[merged["Scenario Name"] == "Actual"]
    test = merged[merged["Scenario Name"] == "Supervisory Severely Adverse"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = train["Date"],
            y = train[col],
            marker_color = 'blue',
            name = "Train"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = test["Date"],
            y = test[col],
            marker_color = 'red',
            name = "Severely Adverse"
        )
    )

    return fig.update_layout(title = "Macro Economic Variable", xaxis_title = "Date", yaxis_title = "", template = 'seaborn', paper_bgcolor = 'rgba(0, 0, 0, 0)')

@callback(Output('autoregressive-model', 'figure'), Input('dropdown-selection', 'value'))
def fit_model(col):
    """
    
    """

    train = merged[merged["Scenario Name"] == "Actual"]

    X_train = pd.DataFrame()
    X_train[col] = train[col]
    X_train['lag_1'] = train[col].shift(1)
    X_train['lag_2'] = train[col].shift(2)
    X_train['lag_3'] = train[col].shift(3)
    X_train['lag_4'] = train[col].shift(4)
    X_train = X_train.pct_change(1).dropna()

    test = merged[merged["Scenario Name"] == "Supervisory Severely Adverse"]

    X_test = pd.DataFrame()
    X_test[col] = test[col]
    X_test['lag_1'] = test[col].shift(1)
    X_test['lag_2'] = test[col].shift(2)
    X_test['lag_3'] = test[col].shift(3)
    X_test['lag_4'] = test[col].shift(4)
    X_test = X_test.pct_change(1).dropna()

    y_train =  merged[merged["Scenario Name"] == "Actual"]["NII"].iloc[4:].pct_change(1).dropna()

    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)

    train_r2 = reg.score(X_train, y_train)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = train["Date"],
            y = train["NII"],
            marker_color = 'blue',
            name = "Net Interest Income"
        )
    )

    tmp = train["NII"].iloc[-1]
    normalized_pred = []

    for i in range(len(pred)):
        tmp += tmp * pred[i]
        normalized_pred.append(tmp)

    fig.add_trace(
        go.Scatter(
            x = test["Date"],
            y = normalized_pred,
            marker_color = 'red',
            name = "Severely Adverse"
        )
    )

    return fig.update_layout(title = "Stress Test Scenario " + str(round(train_r2, 4)), xaxis_title = "Date", yaxis_title = "NET INTEREST INCOME", template = 'seaborn', paper_bgcolor = 'rgba(0, 0, 0, 0)')

if __name__ == '__main__':
    app.run(debug = True)







