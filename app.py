import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("🤖 Herramienta de Trading con Machine Learning")

# --- 1. CONFIGURACIÓN EN LA BARRA LATERAL ---
st.sidebar.header("Parámetros de Configuración")
ACTIVO = st.sidebar.text_input("Símbolo del Activo", value="AAPL")
TIMEFRAME = st.sidebar.selectbox("Timeframe", ['1d', '1h', '5m'], index=0)

# --- MEJORA CLAVE: Selección de período dinámica para evitar errores de descarga ---
# El período de datos se ajusta automáticamente según el timeframe para no exceder los límites de la API.
period_options = {
    '5m': ['60d', '6mo'],
    '1h': ['60d', '6mo', '1y'],
    '1d': ['1y', '2y', '5y']
}
valid_periods = period_options.get(TIMEFRAME, ['1y'])
PERIODO = st.sidebar.selectbox("Período de Datos", valid_periods, index=0)

# --- MEJORA: Mensaje de ayuda actualizado ---
st.sidebar.info("💡 **Tip:** El 'Período de Datos' se ajusta automáticamente según el 'Timeframe' para evitar errores de descarga. Para datos de 5m, el máximo es 6 meses.")

# --- NUEVO: MODO DE OPERACIÓN ---
st.sidebar.header("Modo de Operación")
MODO_BINARIAS = st.sidebar.checkbox("Activar Modo Opciones Binarias", value=False, help="Activa el simulador de Opciones Binarias con expiración fija.")

# Parámetros del Backtester (se usan en modo tradicional)
st.sidebar.header("Parámetros del Backtester (Modo Tradicional)")
STOP_LOSS_PCT = st.sidebar.slider("Stop-Loss (%)", 1, 20, 5) / 100
TAKE_PROFIT_PCT = st.sidebar.slider("Take-Profit (%)", 1, 30, 10) / 100
USE_TRAILING_STOP = st.sidebar.checkbox("Usar Trailing Stop", value=True)
TRAILING_STOP_PCT = st.sidebar.slider("Trailing Stop (%)", 1, 20, 3) / 100

# --- NUEVO: Parámetros para Opciones Binarias ---
if MODO_BINARIAS:
    st.sidebar.subheader("Parámetros de Opciones Binarias")
    EXPIRACION_MINUTOS = st.sidebar.selectbox("Tiempo de Expiración", [5, 15, 30, 60, 240], index=0, help="Tiempo hasta que la opción caduca.")
    PAYOUT_PCT = st.sidebar.slider("Porcentaje de Pago (%)", 70, 95, 85) / 100
    INVERSION_POR_OPERACION = st.sidebar.number_input("Inversión por Operación ($)", value=100, min_value=1)

# Estrategia a utilizar
st.sidebar.header("Estrategia de Trading")
ESTRATEGIA = st.sidebar.selectbox("Selecciona Estrategia", 
                                  ['Momentum', 'Mean Reversion', 'MACD Crossover', 'Stochastic Oscillator', 'VWAP Trading', 'Machine Learning (RF)'], 
                                  index=5)

# Parámetros para la estrategia de ML
ML_THRESHOLD = 0.6
if ESTRATEGIA == 'Machine Learning (RF)':
    st.sidebar.subheader("Parámetros de ML")
    ML_THRESHOLD = st.sidebar.slider("Umbral de Confianza para Comprar (%)", 50, 90, 60) / 100

# --- 2. OBTENER DATOS Y CALCULAR INDICADORES ---
# --- MEJORA: Función de carga de datos mucho más robusta ---
@st.cache_data
def cargar_datos_robusto(activo, periodo, intervalo):
    """
    Función robusta para descargar datos, intentando con símbolos alternativos
    y manejando errores comunes de yfinance.
    """
    simbolos_a_probar = [activo]
    if activo.endswith('-USD'):
        simbolos_a_probar.append(activo.replace('-USD', '=X'))

    for simbolo in simbolos_a_probar:
        with st.spinner(f'Descargando datos para {simbolo}...'):
            try:
                datos = yf.download(simbolo, period=periodo, interval=intervalo, progress=False)
                
                if datos.empty:
                    st.warning(f"⚠️ No se encontraron datos para '{simbolo}' con la configuración actual (Período: {periodo}, Intervalo: {intervalo}).")
                    continue

                if isinstance(datos.columns, pd.MultiIndex):
                    datos.columns = ['_'.join(col).strip() for col in datos.columns.values]
                    rename_dict = {f'Open_{simbolo}': 'Open', f'High_{simbolo}': 'High', f'Low_{simbolo}': 'Low',
                                   f'Close_{simbolo}': 'Close', f'Adj Close_{simbolo}': 'Adj Close', f'Volume_{simbolo}': 'Volume'}
                    datos = datos.rename(columns=rename_dict)
                
                st.success(f"✅ Datos descargados exitosamente para '{simbolo}'.")
                return datos

            except Exception as e:
                st.warning(f"⚠️ Error al descargar '{simbolo}': {e}. Intentando alternativa...")
                continue
    
    st.error(f"❌ No se pudieron descargar los datos para ningún símbolo de la lista: {simbolos_a_probar}.")
    st.info("💡 **Posibles soluciones:**")
    st.info("- Revisa que el símbolo del activo sea correcto (ej. 'AAPL', 'BTC-USD', 'EURUSD=X').")
    st.info("- Si pides datos intradía (ej. 5m), el período no puede ser muy largo (ej. '2y'). Prueba con '60d' o '6mo'.")
    st.info("- Asegúrate de tener conexión a internet.")
    return None

datos_historicos = cargar_datos_robusto(ACTIVO, PERIODO, TIMEFRAME)

if datos_historicos is not None:
    # --- 3. APLICAR ALGORITMOS Y SEÑALES ---
    datos_historicos.ta.ema(length=20, append=True)
    datos_historicos.ta.ema(length=50, append=True)
    datos_historicos.ta.rsi(length=14, append=True)
    datos_historicos.ta.macd(fast=12, slow=26, signal=9, append=True)
    datos_historicos.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)
    datos_historicos.ta.vwap(append=True)
    datos_historicos.ta.atr(length=14, append=True)
    
    bb_length = 20
    bb_std = 2
    datos_historicos['BBM'] = datos_historicos['Close'].rolling(window=bb_length).mean()
    datos_historicos['BBL'] = datos_historicos['BBM'] - (datos_historicos['Close'].rolling(window=bb_length).std() * bb_std)
    datos_historicos['BBU'] = datos_historicos['BBM'] + (datos_historicos['Close'].rolling(window=bb_length).std() * bb_std)
    
    datos_historicos['Volume_SMA'] = datos_historicos['Volume'].rolling(window=20).mean()
    datos_historicos['volumen_alto'] = datos_historicos['Volume'] > datos_historicos['Volume_SMA'] * 1.2

    # --- Definición de Señales para cada Estrategia ---
    datos_historicos['tendencia_alcista'] = (datos_historicos['Close'] > datos_historicos['EMA_20']) & (datos_historicos['Close'] > datos_historicos['EMA_50'])
    datos_historicos['senal_momentum'] = (
        (datos_historicos['Close'] > datos_historicos['EMA_20']) & 
        (datos_historicos['Close'].shift(1) <= datos_historicos['EMA_20']) & 
        datos_historicos['tendencia_alcista'] & 
        (datos_historicos['RSI_14'] < 70) &
        datos_historicos['volumen_alto']
    )
    datos_historicos['senal_mean_reversion'] = (
        (datos_historicos['Close'] < datos_historicos['BBL']) & 
        (datos_historicos['RSI_14'] < 30) &
        datos_historicos['volumen_alto']
    )
    datos_historicos['senal_macd'] = (
        (datos_historicos['MACD_12_26_9'] > datos_historicos['MACDs_12_26_9']) &
        (datos_historicos['MACD_12_26_9'].shift(1) <= datos_historicos['MACDs_12_26_9'].shift(1)) &
        (datos_historicos['MACD_12_26_9'] < 0)
    )
    datos_historicos['senal_stoch'] = (
        (datos_historicos['STOCHk_14_3_3'] < 20) & 
        (datos_historicos['STOCHk_14_3_3'] > datos_historicos['STOCHd_14_3_3'])
    )
    datos_historicos['senal_vwap'] = (
        (datos_historicos['Close'] < datos_historicos['VWAP_D']) &
        (datos_historicos['Close'].shift(1) >= datos_historicos['VWAP_D'].shift(1)) &
        datos_historicos['volumen_alto']
    )
    datos_historicos['senal_venta'] = (
        (datos_historicos['Close'] < datos_historicos['EMA_20']) &
        (datos_historicos['Close'].shift(1) >= datos_historicos['EMA_20'].shift(1)) &
        (datos_historicos['RSI_14'] > 30)
    )

    # Estrategia 6: Machine Learning (Random Forest)
    datos_historicos['senal_ml'] = False
    if ESTRATEGIA == 'Machine Learning (RF)':
        st.subheader("🧠 Entrenando Modelo de Machine Learning...")
        
        if MODO_BINARIAS:
            timeframe_minutes = int(TIMEFRAME.replace('m', '').replace('h', '60').replace('d', '1440'))
            expiracion_steps = EXPIRACION_MINUTOS // timeframe_minutes
            prediction_horizon = expiracion_steps
        else:
            prediction_horizon = 5

        if prediction_horizon < 1:
            st.error(f"⚠️ Error de Configuración: El tiempo de expiración ({EXPIRACION_MINUTOS} min) es menor que el timeframe de los datos ({TIMEFRAME}). No se puede entrenar el modelo.")
        else:
            features = ['EMA_20', 'EMA_50', 'RSI_14', 'ATRr_14', 'Volume_SMA', 'MACD_12_26_9', 'MACDh_12_26_9', 'STOCHk_14_3_3', 'VWAP_D']
            df_ml = datos_historicos[features].copy()
            df_ml.dropna(inplace=True)

            df_ml['target'] = datos_historicos['Close'].shift(-prediction_horizon) > datos_historicos['Close']
            df_ml.dropna(inplace=True)
            
            if df_ml['target'].nunique() < 2:
                st.warning("⚠️ Advertencia: La variable objetivo para el modelo solo contiene una clase. No se puede entrenar un modelo de clasificación útil con estos parámetros.")
            else:
                X = df_ml.drop('target', axis=1)
                y = df_ml['target']
                split_index = int(len(X) * 0.8)
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
                model.fit(X_train, y_train)
                st.success("✅ Modelo entrenado con éxito.")

                probabilidades = model.predict_proba(df_ml[features])[:, 1]
                
                df_ml['probabilidad_subida'] = probabilidades
                df_ml['senal_ml_pred'] = df_ml['probabilidad_subida'] > ML_THRESHOLD
                
                datos_historicos = datos_historicos.join(df_ml[['senal_ml_pred', 'probabilidad_subida']], how='left')
                datos_historicos['senal_ml'] = datos_historicos['senal_ml_pred'].fillna(False)
                
                accuracy = model.score(X_test, y_test)
                st.info(f"Precisión del modelo en datos de prueba: {accuracy:.2f}")

    # --- Seleccionar la estrategia final ---
    if ESTRATEGIA == 'Momentum':
        datos_historicos['senal_compra'] = datos_historicos['senal_momentum']
    elif ESTRATEGIA == 'Mean Reversion':
        datos_historicos['senal_compra'] = datos_historicos['senal_mean_reversion']
    elif ESTRATEGIA == 'MACD Crossover':
        datos_historicos['senal_compra'] = datos_historicos['senal_macd']
    elif ESTRATEGIA == 'Stochastic Oscillator':
        datos_historicos['senal_compra'] = datos_historicos['senal_stoch']
    elif ESTRATEGIA == 'VWAP Trading':
        datos_historicos['senal_compra'] = datos_historicos['senal_vwap']
    else:
        datos_historicos['senal_compra'] = datos_historicos['senal_ml']

    # --- 4. BACKTESTER ---
    resultados_operaciones = []
    
    if MODO_BINARIAS:
        st.header("Resultados del Backtester (Modo Opciones Binarias)")
        en_posicion = False
        precio_ejecucion = 0
        fecha_ejecucion = None
        direccion_prediccion = None
        
        timeframe_minutes = int(TIMEFRAME.replace('m', '').replace('h', '60').replace('d', '1440'))
        expiracion_steps = EXPIRACION_MINUTOS // timeframe_minutes

        for i in range(len(datos_historicos) - expiracion_steps):
            if not en_posicion and datos_historicos.iloc[i]['senal_compra']:
                en_posicion = True
                precio_ejecucion = datos_historicos.iloc[i]['Close']
                fecha_ejecucion = datos_historicos.index[i]
                direccion_prediccion = 'CALL'
            elif not en_posicion and datos_historicos.iloc[i]['senal_venta']:
                 en_posicion = True
                 precio_ejecucion = datos_historicos.iloc[i]['Close']
                 fecha_ejecucion = datos_historicos.index[i]
                 direccion_prediccion = 'PUT'
            elif en_posicion:
                fecha_expiracion = datos_historicos.index[i + expiracion_steps]
                precio_en_expiracion = datos_historicos.iloc[i + expiracion_steps]['Close']
                
                gano = False
                if direccion_prediccion == 'CALL' and precio_en_expiracion > precio_ejecucion:
                    gano = True
                elif direccion_prediccion == 'PUT' and precio_en_expiracion < precio_ejecucion:
                    gano = True
                
                rentabilidad = PAYOUT_PCT if gano else -1.0
                
                resultados_operaciones.append({
                    'fecha_ejecucion': fecha_ejecucion, 'fecha_expiracion': fecha_expiracion,
                    'direccion': direccion_prediccion, 'precio_ejecucion': precio_ejecucion,
                    'precio_expiracion': precio_en_expiracion, 'resultado': 'GANA' if gano else 'PIERDE',
                    'rentabilidad': rentabilidad, 'beneficio': INVERSION_POR_OPERACION * rentabilidad
                })
                en_posicion = False
    else:
        st.header("Resultados del Backtester (Modo Tradicional)")
        en_posicion = False
        precio_entrada = 0
        stop_loss_actual = 0
        take_profit_actual = 0
        
        for i in range(1, len(datos_historicos)):
            fila_actual = datos_historicos.iloc[i]
            
            if not en_posicion and fila_actual['senal_compra']:
                en_posicion = True
                precio_entrada = fila_actual['Close']
                atr_actual = fila_actual['ATRr_14']
                stop_loss_actual = precio_entrada * (1 - max(STOP_LOSS_PCT, atr_actual * 2 if not np.isnan(atr_actual) else STOP_LOSS_PCT))
                take_profit_actual = precio_entrada * (1 + TAKE_PROFIT_PCT)
                
            elif en_posicion:
                precio_salida = 0
                if USE_TRAILING_STOP:
                    nuevo_trailing_stop = fila_actual['Close'] * (1 - TRAILING_STOP_PCT)
                    if nuevo_trailing_stop > stop_loss_actual:
                        stop_loss_actual = nuevo_trailing_stop
                
                if fila_actual['High'] >= take_profit_actual:
                    precio_salida = take_profit_actual
                elif fila_actual['Low'] <= stop_loss_actual:
                    precio_salida = stop_loss_actual
                
                if precio_salida > 0:
                    rentabilidad = (precio_salida - precio_entrada) / precio_entrada
                    resultados_operaciones.append({
                        'entrada': precio_entrada, 'salida': precio_salida, 'rentabilidad': rentabilidad,
                        'fecha_entrada': datos_historicos.index[i-1], 'fecha_salida': datos_historicos.index[i]
                    })
                    en_posicion = False

    # --- 5. MOSTRAR RESULTADOS ---
    if not resultados_operaciones:
        st.warning("No se generaron operaciones en el período seleccionado con los parámetros actuales.")
    else:
        df_operaciones = pd.DataFrame(resultados_operaciones)
        total_ops = len(df_operaciones)
        ops_ganadoras = sum(1 for r in df_operaciones['rentabilidad'] if r > 0)
        porcentaje_aciertos = (ops_ganadoras / total_ops) * 100
        
        if MODO_BINARIAS:
            beneficio_total = df_operaciones['beneficio'].sum()
            inversion_total = total_ops * INVERSION_POR_OPERACION
            roi_total = (beneficio_total / inversion_total) * 100 if inversion_total > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Operaciones", total_ops)
            col2.metric("% Aciertos", f"{porcentaje_aciertos:.2f}%")
            col3.metric("Beneficio Neto", f"${beneficio_total:.2f}", delta=f"{roi_total:.2f}%")
            
            with st.expander("Ver Detalles de Operaciones Binarias"):
                st.dataframe(df_operaciones)
        else:
            rentabilidad_total = df_operaciones['rentabilidad'].sum()
            rentabilidad_media = df_operaciones['rentabilidad'].mean()
            max_ganancia = df_operaciones['rentabilidad'].max()
            max_perdida = df_operaciones['rentabilidad'].min()
            factor_beneficio = abs(df_operaciones[df_operaciones['rentabilidad'] > 0]['rentabilidad'].sum() / df_operaciones[df_operaciones['rentabilidad'] < 0]['rentabilidad'].sum()) if not df_operaciones[df_operaciones['rentabilidad'] < 0].empty else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Operaciones", total_ops)
            col2.metric("Ops. Ganadoras", ops_ganadoras)
            col3.metric("% Aciertos", f"{porcentaje_aciertos:.2f}%")
            col4.metric("Rentabilidad Total", f"{rentabilidad_total * 100:.2f}%", delta=f"{rentabilidad_total * 100:.2f}%")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Rentabilidad Media", f"{rentabilidad_media * 100:.2f}%")
            col6.metric("Máx. Ganancia", f"{max_ganancia * 100:.2f}%")
            col7.metric("Máx. Pérdida", f"{max_perdida * 100:.2f}%")
            col8.metric("Factor Beneficio", f"{factor_beneficio:.2f}")
            
            with st.expander("Ver Detalles de Operaciones"):
                st.dataframe(df_operaciones)

    # --- 6. VISUALIZACIÓN ---
    st.header(f"Gráfico de {ACTIVO}")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        subplot_titles=('Precio', 'RSI / Stoch', 'MACD', 'Volumen'), 
                        row_heights=[0.5, 0.2, 0.2, 0.1])
    
    fig.add_trace(go.Candlestick(x=datos_historicos.index, open=datos_historicos['Open'], high=datos_historicos['High'], low=datos_historicos['Low'], close=datos_historicos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['VWAP_D'], line=dict(color='purple', width=1, dash='dash'), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos[datos_historicos['senal_compra']].index, y=datos_historicos[datos_historicos['senal_compra']]['Close'], mode='markers', marker_symbol='triangle-up', marker_size=12, marker_color='lime', name='Señal Compra (CALL)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos[datos_historicos['senal_venta']].index, y=datos_historicos[datos_historicos['senal_venta']]['Close'], mode='markers', marker_symbol='triangle-down', marker_size=12, marker_color='red', name='Señal Venta (PUT)'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['RSI_14'], line=dict(color='purple'), name='RSI 14'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['STOCHk_14_3_3'], line=dict(color='blue'), name='Stoch %K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['STOCHd_14_3_3'], line=dict(color='red'), name='Stoch %D'), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['MACD_12_26_9'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['MACDs_12_26_9'], line=dict(color='red'), name='Señal MACD'), row=3, col=1)
    fig.add_trace(go.Bar(x=datos_historicos.index, y=datos_historicos['MACDh_12_26_9'], name='Histograma MACD', marker_color='gray'), row=3, col=1)

    fig.add_trace(go.Bar(x=datos_historicos.index, y=datos_historicos['Volume'], name='Volumen', marker_color='lightblue'), row=4, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=1400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
