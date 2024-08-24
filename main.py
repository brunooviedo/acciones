import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier

# Función para descargar datos y calcular indicadores técnicos
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['Signal'] = df['SMA_50'] > df['SMA_200']  # Señal de cruce dorado
        return df.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()

# Función para predecir si el precio subirá utilizando un modelo básico
def predict_stock(df):
    features = ['SMA_50', 'SMA_200', 'RSI']
    X = df[features]
    y = df['Signal'].shift(-1).fillna(False)
    
    model = RandomForestClassifier()
    model.fit(X[:-1], y[:-1])
    prediction = model.predict(X[-1:])[0]
    return prediction

# Configuración de la aplicación en Streamlit
st.title("Predicción de Acciones")

tickers = st.text_input("Ingresa los tickers de las acciones separados por comas", "AAPL,MSFT,GOOGL")
tickers = tickers.split(',')

if st.button("Predecir"):
    st.write("Analizando y prediciendo...")
    for ticker in tickers:
        df = get_stock_data(ticker.strip())
        if df.empty:
            continue
        
        prediction = predict_stock(df)
        
        st.subheader(f"Análisis de {ticker}")
        st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
        
        if prediction:
            st.success(f"¡Se espera que el precio de {ticker} suba!")
        else:
            st.warning(f"No se espera un aumento en el precio de {ticker}.")
