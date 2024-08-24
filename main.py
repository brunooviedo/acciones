import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Función para descargar datos y calcular indicadores técnicos
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
        df['Volume'] = df['Volume']
        df['Target'] = df['Close'].shift(-1) > df['Close']  # Objetivo de predicción (subida del precio)
        return df.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()

# Función para predecir si el precio subirá utilizando un modelo de RandomForest
def predict_stock(df):
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Volume']
    X = df[features]
    y = df['Target']
    
    # Verificar distribución de Target
    st.write(df['Target'].value_counts())
    
    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Ajuste de hiperparámetros para RandomForest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Mostrar el informe de clasificación
    st.write(classification_report(y_test, y_pred))
    
    # Realizar la predicción
    last_features = X_scaled[-1:].reshape(1, -1)
    prediction = best_model.predict(last_features)[0]
    return prediction

# Configuración de la aplicación en Streamlit
st.title("Predicción de Acciones")

tickers = [
    "JEPI", "TSLA", "MAIN", "AMZN", "JEPQ", "VOO", "PG", "MCD", "TQQQ",
    "SCHD", "HD", "SVOL", "QQQ", "VIG", "MSFT", "SOXL", "GOOGL", "AMD",
    "SCHG", "PEP", "LLY", "V", "CELH", "AAPL", "YINN"
]

if st.button("Predecir"):
    st.write("Analizando y prediciendo...")
    for ticker in tickers:
        df = get_stock_data(ticker.strip())
        if df.empty:
            st.write(f"No se pudieron obtener datos para {ticker}.")
            continue
        
        prediction = predict_stock(df)
        
        st.subheader(f"Análisis de {ticker}")
        st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
        
        # Visualización de RSI
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='RSI', ax=ax, color='blue', label='RSI')
        ax.axhline(y=70, color='red', linestyle='--', label='Overbought')
        ax.axhline(y=30, color='green', linestyle='--', label='Oversold')
        ax.set_title(f'RSI de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('RSI')
        ax.legend()
        st.pyplot(fig)
        
        # Visualización de MACD
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='MACD', ax=ax, color='blue', label='MACD')
        sns.lineplot(data=df, x=df.index, y='MACD_Signal', ax=ax, color='orange', label='MACD Signal')
        ax.bar(df.index, df['MACD_Histogram'], color='grey', alpha=0.5, label='MACD Histogram')
        ax.set_title(f'MACD de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('MACD')
        ax.legend()
        st.pyplot(fig)
        
        # Mostrar resultado de la predicción
        if prediction:
            st.success(f"¡Se espera que el precio de {ticker} suba!")
        else:
            st.warning(f"No se espera un aumento en el precio de {ticker}.")
