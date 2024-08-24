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
import numpy as np

# Función para descargar datos y calcular indicadores técnicos
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='5y', interval='1d', progress=False)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
        df['Volume'] = df['Volume']
        
        # Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        
        # Estocástico
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stochastic'] = stoch.stoch()
        df['Stochastic_Signal'] = stoch.stoch_signal()
        
        df['Target'] = df['Close'].shift(-1) > df['Close']  # Objetivo de predicción (subida del precio)
        return df.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()

# Función para predecir si el precio subirá utilizando un modelo de RandomForest
def predict_stock(df):
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Volume', 'BB_High', 'BB_Low', 'Stochastic', 'Stochastic_Signal']
    X = df[features]
    y = df['Target']
    
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
    st.write("Informe de clasificación:")
    st.text(classification_report(y_test, y_pred))
    
    # Realizar la predicción
    last_features = X_scaled[-1:].reshape(1, -1)
    prediction = best_model.predict(last_features)[0]
    return prediction, df

# Función para obtener una lista de tickers de ejemplo
def get_example_tickers():
    # Lista de tickers populares o de un índice amplio como el S&P 500
    example_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NKE", "NVDA", "IBM", "BA"
    ]
    return example_tickers

# Configuración de la aplicación en Streamlit
st.title("Predicción de Acciones")

# Entrada de tickers del usuario
user_tickers = st.text_input("Ingresa los tickers de las acciones separados por comas (deja en blanco para usar ejemplos)", "")
user_tickers = [ticker.strip().upper() for ticker in user_tickers.split(',') if ticker.strip()]

# Lista de tickers de ejemplo si el usuario no ingresa ninguno
if not user_tickers:
    tickers = get_example_tickers()
else:
    tickers = user_tickers

if st.button("Predecir"):
    st.write("Analizando y prediciendo...")
    for ticker in tickers:
        df = get_stock_data(ticker)
        if df.empty:
            st.write(f"No se pudieron obtener datos para {ticker}.")
            continue
        
        prediction, df = predict_stock(df)
        
        st.subheader(f"Análisis de {ticker}")
        
        # Gráficos de tendencias
        st.write("### Gráfico de Precios y Medias Móviles")
        st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
        
        # Gráfico de Bandas de Bollinger
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='Close', ax=ax, color='blue', label='Precio de Cierre')
        ax.fill_between(df.index, df['BB_High'], df['BB_Low'], color='grey', alpha=0.2, label='Bandas de Bollinger')
        ax.set_title(f'Bandas de Bollinger de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.legend()
        st.pyplot(fig)
        
        # Gráfico de Estocástico
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='Stochastic', ax=ax, color='blue', label='Estocástico')
        sns.lineplot(data=df, x=df.index, y='Stochastic_Signal', ax=ax, color='orange', label='Señal Estocástica')
        ax.axhline(y=80, color='red', linestyle='--', label='Sobrecompra')
        ax.axhline(y=20, color='green', linestyle='--', label='Sobreventa')
        ax.set_title(f'Estocástico de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Estocástico')
        ax.legend()
        st.pyplot(fig)
        
        # Gráfico de RSI
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='RSI', ax=ax, color='blue', label='RSI')
        ax.axhline(y=70, color='red', linestyle='--', label='Sobrecompra')
        ax.axhline(y=30, color='green', linestyle='--', label='Sobreventa')
        ax.set_title(f'RSI de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('RSI')
        ax.legend()
        st.pyplot(fig)
        
        # Gráfico de MACD
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='MACD', ax=ax, color='blue', label='MACD')
        sns.lineplot(data=df, x=df.index, y='MACD_Signal', ax=ax, color='orange', label='MACD Signal')
        ax.bar(df.index, df['MACD_Histogram'], color='grey', alpha=0.5, label='MACD Histogram')
        ax.set_title(f'MACD de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('MACD')
        ax.legend()
        st.pyplot(fig)
        
        # Gráfico de Volumen
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=df.index, y='Volume', ax=ax, color='purple', label='Volumen')
        ax.set_title(f'Volumen de {ticker}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Volumen')
        ax.legend()
        st.pyplot(fig)
        
        # Mostrar resultado de la predicción
        if prediction:
            st.success(f"¡Se espera que el precio de {ticker} suba!")
        else:
            st.warning(f"No se espera un aumento en el precio de {ticker}.")
        
        # Explicación de la decisión
        st.write("### Explicación de la Decisión")
        
        if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
            st.write("La media móvil de 50 días está por encima de la de 200 días, indicando una posible tendencia alcista.")
        else:
            st.write("La media móvil de 50 días está por debajo de la de 200 días, indicando una posible tendencia bajista.")
        
        if df['RSI'].iloc[-1] > 70:
            st.write("El RSI está en sobrecompra, lo que podría indicar una posible reversión a la baja.")
        elif df['RSI'].iloc[-1] < 30:
            st.write("El RSI está en sobreventa, lo que podría indicar una posible reversión al alza.")
        
        if df['Stochastic'].iloc[-1] > df['Stochastic_Signal'].iloc[-1]:
            st.write("El estocástico está en alza, lo que podría indicar un movimiento alcista.")
        else:
            st.write("El estocástico está en baja, lo que podría indicar un movimiento bajista.")
        
        st.write("---")
