import streamlit as st
import pandas as pd

# Título de la aplicación
st.title('Calculadora de Rentabilidad con y sin Dividendos')

# DataFrame para almacenar las compras
if 'compras' not in st.session_state:
    st.session_state['compras'] = pd.DataFrame(columns=['Fecha', 'Acción', 'Cantidad', 'Precio', 'Dividendos'])

# Formulario para ingresar una nueva compra
with st.form('nueva_compra'):
    st.write("Agregar Nueva Compra")
    fecha = st.date_input('Fecha de compra')
    accion = st.text_input('Nombre de la acción')
    cantidad = st.number_input('Cantidad comprada', min_value=0.0, format="%.2f")
    precio = st.number_input('Precio de compra por acción', min_value=0.0, format="%.2f")
    dividendos = st.number_input('Dividendos por acción (anual)', min_value=0.0, format="%.2f")
    submit = st.form_submit_button('Agregar Compra')

# Almacenar la nueva compra en el DataFrame
if submit:
    nueva_compra = pd.DataFrame({
        'Fecha': [fecha],
        'Acción': [accion],
        'Cantidad': [cantidad],
        'Precio': [precio],
        'Dividendos': [dividendos]
    })
    st.session_state['compras'] = pd.concat([st.session_state['compras'], nueva_compra], ignore_index=True)

# Mostrar las compras agregadas
st.write("Compras realizadas:")
st.write(st.session_state['compras'])

# Cálculo de la rentabilidad
if not st.session_state['compras'].empty:
    total_invertido = (st.session_state['compras']['Cantidad'] * st.session_state['compras']['Precio']).sum()
    total_dividendos = (st.session_state['compras']['Cantidad'] * st.session_state['compras']['Dividendos']).sum()
    
    st.write(f"Total invertido: ${total_invertido:,.2f} CLP")
    st.write(f"Total dividendos recibidos (anual): ${total_dividendos:,.2f} CLP")
    
    # Rentabilidad sin dividendos
    st.subheader("Rentabilidad sin dividendos:")
    precio_actual = st.number_input('Precio actual de la acción', min_value=0.0, format="%.2f")
    if precio_actual > 0:
        valor_actual = (st.session_state['compras']['Cantidad'] * precio_actual).sum()
        rentabilidad_sin_dividendos = ((valor_actual - total_invertido) / total_invertido) * 100
        st.write(f"Rentabilidad sin dividendos: {rentabilidad_sin_dividendos:.2f}%")
    
    # Rentabilidad con dividendos
    st.subheader("Rentabilidad con dividendos:")
    rentabilidad_con_dividendos = ((valor_actual + total_dividendos - total_invertido) / total_invertido) * 100
    st.write(f"Rentabilidad con dividendos: {rentabilidad_con_dividendos:.2f}%")
else:
    st.write("No hay compras registradas.")

# Agregar botón para reiniciar las compras
if st.button('Reiniciar compras'):
    st.session_state['compras'] = pd.DataFrame(columns=['Fecha', 'Acción', 'Cantidad', 'Precio', 'Dividendos'])
    st.write("Las compras han sido reiniciadas.")
