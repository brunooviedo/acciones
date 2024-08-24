import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import xgboost as xgb

# Función para descargar datos y calcular indicadores técnicos
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['Stochastic'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Signal'] = (df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 30)  # Combina señales
        return df.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()

# Función para crear el modelo y realizar predicciones
def train_and_predict(df):
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Stochastic']
    X = df[features]
    y = df['Signal'].shift(-1).fillna(False).astype(int)  # Convertir a enteros
    
    # Revisar el balance de clases en y
    st.write("Balance de clases en y:", y.value_counts())

    # Si las clases están desequilibradas, hacer sobremuestreo
    if y.value_counts().min() == 0:
        st.error("No hay suficiente variabilidad en las clases para entrenar el modelo.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.3, random_state=42)

    # Sobremuestrear la clase minoritaria si es necesario
    if y_train.value_counts()[0] > y_train.value_counts()[1]:
        X_train, y_train = resample(X_train[y_train == 1],
                                    y_train[y_train == 1],
                                    replace=True,
                                    n_samples=y_train.value_counts()[0],
                                    random_state=42)
    else:
        X_train, y_train = resample(X_train[y_train == 0],
                                    y_train[y_train == 0],
                                    replace=True,
                                    n_samples=y_train.value_counts()[1],
                                    random_state=42)
    
    # Modelos Ensemble
    models = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': xgb.XGBClassifier()
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"{name} Accuracy: {accuracy:.2f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        except Exception as e:
            st.error(f"Error al entrenar {name}: {e}")
    
    st.write("Mejor modelo:", type(best_model).__name__)
    
    # Predicción con el mejor modelo
    if best_model:
        prediction = best_model.predict(X[-1:])[0]
        return prediction
    else:
        return None

# Configuración de la aplicación en Streamlit
st.title("Predicción de Acciones Mejorada")

tickers = st.text_input("Ingresa los tickers de las acciones separados por comas", "AAPL,MSFT,GOOGL")
tickers = tickers.split(',')

if st.button("Predecir"):
    st.write("Analizando y prediciendo...")
    for ticker in tickers:
        df = get_stock_data(ticker.strip())
        if df.empty:
            continue
        
        prediction = train_and_predict(df)
        
        st.subheader(f"Análisis de {ticker}")
        st.line_chart(df[['Close', 'SMA_50', 'SMA_200']])
        
        if prediction:
            st.success(f"¡Se espera que el precio de {ticker} suba!")
        else:
            st.warning(f"No se espera un aumento en el precio de {ticker}.")
