import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("⚡ Germany Electricity Consumption Forecast (Simple Moving Average)")

# === Загружаем данные ===
data_path = "C:/Users/user/streamlit-crash-cours/data/opsd_germany_daily.csv"
df = pd.read_csv(data_path)

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

series = df["Consumption"].dropna()

st.subheader("Данные (первые строки):")
st.write(series.head())

# === Настройки ===
window = st.sidebar.slider("Окно скользящего среднего (дней):", 3, 30, 7)

# === Простая модель: прогноз = среднее за последние N дней ===
last_avg = series.iloc[-window:].mean()
forecast = pd.Series([last_avg] * 7,
                     index=pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=7, freq="D"))

st.subheader("Прогноз на неделю (простая модель):")
st.write(forecast)

# === 1. График: История + прогноз ===
fig, ax = plt.subplots(figsize=(12, 5))
series.plot(ax=ax, label="Исторические данные")
forecast.plot(ax=ax, label="Прогноз", color="red")
plt.legend()
st.pyplot(fig)

# === 2. График: Скользящее среднее ===
rolling_series = series.rolling(window=window).mean()

fig2, ax2 = plt.subplots(figsize=(12, 4))
series.plot(ax=ax2, alpha=0.5, label="Исторические данные")
rolling_series.plot(ax=ax2, color="orange", label=f"Скользящее среднее ({window} дней)")
plt.legend()
st.pyplot(fig2)