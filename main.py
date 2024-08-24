import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Función para calcular el ADL manualmente
def calculate_adl(df):
    df['ADL'] = 0
    for i in range(1, len(df)):
        df.loc[df.index[i], 'ADL'] = df.loc[df.index[i-1], 'ADL'] + \
            ((2 * df.loc[df.index[i], 'Close'] - df.loc[df.index[i], 'Low'] - df.loc[df.index[i], 'High']) / 
            (df.loc[df.index[i], 'High'] - df.loc[df.index[i], 'Low'])) * df.loc[df.index[i], 'Volume']
    return df

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
        
        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # CCI
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Calcular ADL manualmente
        df = calculate_adl(df)
        
        df['Target'] = df['Close'].shift(-1) > df['Close']  # Objetivo de predicción (subida del precio)
        
        # Verifica el balance de las clases
        st.write(f"Balance de clases para {ticker}:")
        st.write(df['Target'].value_counts())
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()

# Función para predecir si el precio subirá utilizando un modelo de GradientBoosting
def predict_stock(df):
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Volume', 'BB_High', 'BB_Low', 'Stochastic', 'Stochastic_Signal', 'ATR', 'CCI', 'ADL']
    X = df[features]
    y = df['Target']
    
    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir los datos en entrenamiento y prueba con estratificación
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid={
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    }, cv=skf, scoring='accuracy')
    
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    
    # Evaluar el modelo
    y_pred = best_model.predict(X_scaled)
    
    # Mostrar el informe de clasificación
    st.write("Informe de clasificación:")
    st.text(classification_report(y, y_pred))
    
    # Realizar la predicción
    last_features = X_scaled[-1:].reshape(1, -1)
    prediction = best_model.predict(last_features)[0]
    return prediction, df

# Main Streamlit App
def main():
    st.title("Análisis y Predicción de Acciones")
    
    ticker = st.text_input("Introduce el ticker de la acción (por ejemplo, AAPL):")
    
    if ticker:
        df = get_stock_data(ticker)
        
        if not df.empty:
            st.write("Datos de la acción:")
            st.write(df.tail())
            
            # Graficar los datos y los indicadores técnicos
            st.subheader("Gráficos de Indicadores Técnicos")
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            ax[0, 0].plot(df['Close'], label='Precio de Cierre')
            ax[0, 0].plot(df['SMA_50'], label='SMA 50 días', alpha=0.7)
            ax[0, 0].plot(df['SMA_200'], label='SMA 200 días', alpha=0.7)
            ax[0, 0].set_title('Precio de Cierre y Medias Móviles')
            ax[0, 0].legend()
            
            ax[0, 1].plot(df['RSI'], label='RSI')
            ax[0, 1].set_title('Índice de Fuerza Relativa (RSI)')
            
            ax[1, 0].plot(df['MACD'], label='MACD')
            ax[1, 0].plot(df['MACD_Signal'], label='MACD Signal', alpha=0.7)
            ax[1, 0].set_title('MACD y Línea de Señal')
            ax[1, 0].legend()
            
            ax[1, 1].plot(df['BB_High'], label='Banda Superior')
            ax[1, 1].plot(df['BB_Low'], label='Banda Inferior')
            ax[1, 1].set_title('Bandas de Bollinger')
            ax[1, 1].legend()
            
            st.pyplot(fig)
            
            prediction, df = predict_stock(df)
            st.write(f"La predicción para {ticker} es: {'Subirá' if prediction else 'Bajará'}")

if __name__ == "__main__":
    main()
