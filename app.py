import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Load Data and Saved Model ---
print("Loading data and pre-trained model...")
try:
    lstm_model = load_model("lstm_model.h5")
except OSError:
    print("Model file not found. Please run train_model.py first.")
    exit()

data = pd.read_csv("ngpf_data.csv")
data.rename(columns={'Day': 'date', 'Price in Dollars per Million Btu': 'gas_price'}, inplace=True)
data['date'] = pd.to_datetime(data['date'], format="%d/%m/%Y")
data = data.sort_values(by='date').set_index('date')
data = data.fillna(method='pad')

# --- Data Preparation for Prediction ---
train = data['1997-01-07': '2020-01-06']
test = data['2020-01-07': '2022-03-01']

slot = 15
x_train, y_train = [], []
for i in range(slot, len(train)):
    x_train.append(train.iloc[i-slot:i, 0])
    y_train.append(train.iloc[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

dataset_total = pd.concat((train, test), axis=0)
inputs = dataset_total[len(dataset_total) - len(test) - slot:].values
inputs = inputs.reshape(-1, 1)
x_test = []
for i in range(slot, len(test) + slot):
    x_test.append(inputs[i - slot:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# --- Make Predictions using the loaded model ---
yp_train_lstm = lstm_model.predict(x_train)
train_compare = pd.DataFrame(yp_train_lstm, index=train.iloc[slot:].index, columns=['gp_pred_lstm'])
train_compare['gas_price'] = train.iloc[slot:]['gas_price']

pred_price_lstm = lstm_model.predict(x_test)
test_compare = pd.DataFrame(pred_price_lstm, index=test.index, columns=['gp_pred_lstm'])
test_compare['gas_price'] = test['gas_price']

print("Predictions complete.")

# --- Dash App Setup ---
# Define the CSS styles as a list of strings
# The first link imports the Lato font from Google Fonts.
# The second string contains the CSS rule to apply the font to the body.
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap',
    {
        'href': 'data:text/css;charset=UTF-8,%s' % """
            body { font-family: 'Lato', sans-serif; }
        """
    }
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Natural Gas Price Prediction Dashboard"

# Define the dashboard layout
app.layout = html.Div(style={'backgroundColor': '#F0F2F5', 'color': '#333333'}, children=[
    
    html.H1(
        children='Natural Gas Price Forecasting Dashboard',
        style={'textAlign': 'center', 'color': '#1E3A8A'}
    ),
    html.Div(
        children='''
            This dashboard visualizes historical natural gas prices and a forecast for the near future 
            using a Long Short-Term Memory (LSTM) model.
        ''',
        style={'textAlign': 'center', 'marginBottom': '20px'}
    ),

    dcc.Tabs(id="tabs-graph", value='tab-historical-prices', children=[
        dcc.Tab(label='Historical Prices', value='tab-historical-prices', children=[
            html.Div([
                html.H3('Natural Gas Spot Prices', style={'textAlign': 'center'}),
                dcc.Graph(id='gas-price-graph'),
                html.Div([
                    html.H4('Select Date Range:', style={'textAlign': 'left', 'marginLeft': '2%'}),
                    dcc.RangeSlider(
                        id='date-range-slider',
                        min=0, max=len(data), step=1,
                        value=[0, len(data)],
                        marks={i: {'label': data.index[i].strftime('%Y'), 'style': {'color': '#1E3A8A'}}
                               for i in range(0, len(data), 1000)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ])
        ]),

        dcc.Tab(label='Model Performance', value='tab-model-performance', children=[
            html.Div([
                html.H3('Model Prediction vs. Actual Prices', style={'textAlign': 'center'}),
                dcc.Graph(id='train-test-graph'),
                html.Div(id='model-metrics', style={'textAlign': 'center', 'fontSize': '1.2em', 'marginTop': '20px'})
            ])
        ])
    ]),
])

# Callback for the Historical Prices tab
@app.callback(
    Output('gas-price-graph', 'figure'),
    [Input('date-range-slider', 'value')]
)
def update_historical_graph(date_range):
    start_index, end_index = date_range
    dff = data.iloc[start_index:end_index]
    fig = px.line(dff, x=dff.index, y='gas_price', title='Natural Gas Spot Prices Over Time')
    fig.update_layout(
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', font_color='#333333'
    )
    return fig

# Callback for the Model Performance tab
@app.callback(
    Output('train-test-graph', 'figure'),
    Output('model-metrics', 'children'),
    [Input('tabs-graph', 'value')]
)
def update_model_performance(tab_name):
    if tab_name == 'tab-model-performance':
        # Calculate metrics
        mse_train = mean_squared_error(train_compare['gas_price'], train_compare['gp_pred_lstm'])
        r2_train = r2_score(train_compare['gas_price'], train_compare['gp_pred_lstm'])
        mse_test = mean_squared_error(test_compare['gas_price'], test_compare['gp_pred_lstm'])
        r2_test = r2_score(test_compare['gas_price'], test_compare['gp_pred_lstm'])

        metrics_text = [
            html.P(f"Train Data: MSE = {mse_train:.4f}, R-squared = {r2_train:.4f}"),
            html.P(f"Test Data: MSE = {mse_test:.4f}, R-squared = {r2_test:.4f}")
        ]

        # Plotting the train and test data with predictions
        fig = px.line()
        fig.add_scatter(x=train_compare.index, y=train_compare['gas_price'], mode='lines', name='Train Actual', line=dict(color='blue'))
        fig.add_scatter(x=train_compare.index, y=train_compare['gp_pred_lstm'], mode='lines', name='Train Predicted', line=dict(color='lightgreen', dash='dot'))
        fig.add_scatter(x=test_compare.index, y=test_compare['gas_price'], mode='lines', name='Test Actual', line=dict(color='red'))
        fig.add_scatter(x=test_compare.index, y=test_compare['gp_pred_lstm'], mode='lines', name='Test Predicted', line=dict(color='orange', dash='dot'))

        fig.update_layout(
            title='Actual vs. Predicted Prices (Train and Test Sets)',
            xaxis_title='Date', yaxis_title='Natural Gas Price',
            legend_title='Series', plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', font_color='#333333'
        )
        return fig, metrics_text
    return {}, []

if __name__ == '__main__':
    app.run(debug=True)