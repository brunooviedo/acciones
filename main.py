import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configuración de la página de Streamlit
st.set_page_config(page_title="Radar de Aumento de Acciones", layout="wide")

# Título de la aplicación
st.title("Radar de Aumento de Acciones")

# Configuración del rango de fechas
today = datetime.now().date()
start_date = today - timedelta(days=365)  # Datos del último año

# Interfaz de usuario para ingresar el símbolo de la acción
symbol = st.text_input("Ingrese el símbolo de la acción", "AAPL").upper()

# Obtener los datos históricos
def get_data(symbol, start_date, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            data = yf.download(symbol, start=start_date, end=today)
            if data.empty:
                st.warning("No se encontraron datos para el símbolo ingresado.")
            data['Date'] = data.index
            return data
        except Exception as e:
            attempt += 1
            st.error(f"Error al obtener datos (intento {attempt}): {e}")
            time.sleep(delay)
    st.error("No se pudo obtener datos después de varios intentos.")
    return pd.DataFrame()

data = get_data(symbol, start_date)

if not data.empty:
    # Cálculo de indicadores técnicos
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

    # Condiciones de compra basadas en los indicadores
    last_rsi = data['RSI'].iloc[-1]
    last_ema_20 = data['EMA_20'].iloc[-1]
    last_ema_50 = data['EMA_50'].iloc[-1]

    buy_signal = last_rsi < 30 and last_ema_20 > last_ema_50  # Ejemplo de condiciones para compra

    # Visualización de los datos
    fig = go.Figure()

    # Gráfico de precios
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'))

    # Gráficos de EMA
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50'))

    # Configuración del diseño
    fig.update_layout(title=f'Análisis de {symbol}',
                      xaxis_title='Fecha',
                      yaxis_title='Precio',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

    # Mostrar señales de compra
    if buy_signal:
        st.success(f"¡Señal de compra para {symbol}!")
    else:
        st.warning(f"No hay señal de compra para {symbol}.")
else:
    st.error("No se encontraron datos para el símbolo ingresado.")
