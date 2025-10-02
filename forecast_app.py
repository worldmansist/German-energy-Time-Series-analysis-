import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Title
st.title("Load Forecasting based on RNN")

# Downloading CSV
uploaded_file = st.file_uploader("Upload dfd.csv", type="csv")

if uploaded_file is not None:
    dfd = pd.read_csv(uploaded_file, parse_dates=["Datetime"], index_col="Datetime")

    df_Load = dfd['Load (incl. self-consumption)']
    df_Load.name = "Load"
    df_Load_2023 = df_Load.loc["2022-06-01":]
    dfDP = df_Load_2023.copy().to_frame()

    scaler = MinMaxScaler()
    dfDP['Load'] = scaler.fit_transform(dfDP[['Load']])

    dfDP.index = pd.to_datetime(dfDP.index)

    dfDP['hour'] = dfDP.index.hour
    dfDP['dayofweek'] = dfDP.index.dayofweek
    dfDP['month'] = dfDP.index.month
    dfDP['is_weekend'] = (dfDP.index.dayofweek >= 5).astype(int)

    scaler_ext = MinMaxScaler()
    external_features = scaler_ext.fit_transform(dfDP[['hour','dayofweek','month','is_weekend']])
    dfDP[['hour','dayofweek','month','is_weekend']] = external_features

    feature_cols = ['Load','hour','dayofweek','month','is_weekend']
    ylist = dfDP['Load'].tolist()
    X_features = dfDP[feature_cols].values

    n_future = 96
    n_past = 672
    total_period = n_past + n_future

    X_new, y_new = [], []
    idx_end = len(dfDP)
    idx_start = idx_end - total_period

    while idx_start > 0:
        x_window = X_features[idx_start:idx_start+n_past, :]
        y_window = ylist[idx_start+n_past:idx_start+total_period]
        X_new.append(x_window)
        y_new.append(y_window)
        idx_start -= 1

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=42)
    X_test_rs = X_test

    # Downloading model
    @st.cache_resource
    def load_rnn_model():
        return load_model(r"C:\Users\user\streamlit-crash-cours\models\my_modelRNN.keras")

    modelRNN = load_rnn_model()

    st.subheader("Model name")
    st.write("**my_modelRNN.keras**")

    # Forecast
    preds = modelRNN.predict(X_test_rs)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))
    mae = mean_absolute_error(y_test.flatten(), preds.flatten())
    r2 = r2_score(y_test.flatten(), preds.flatten())

    st.subheader("Metrics of quality")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    # Slider for forecast choosing
    idx = st.slider("Choose forecast index (example from test data)", 0, len(y_test)-1, 0)

    st.subheader(f"График прогноза для тестового примера №{idx}")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_test[idx], label="True values")
    ax.plot(preds[idx], label="Forecast")
    ax.legend()
    st.pyplot(fig)