

¡Claro que sí! Un buen `README.md` es fundamental para cualquier proyecto en GitHub. Aquí te propongo una estructura completa y detallada que puedes usar. Solo tienes que copiar y pegar este contenido en un archivo llamado `README.md` en la raíz de tu repositorio.

---

# 🤖 Herramienta de Trading con Machine Learning y Opciones Binarias

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Una aplicación web interactiva construida con Streamlit para analizar activos financieros, probar estrategias de trading y utilizar un modelo de Machine Learning para predecir movimientos del mercado. La herramienta soporta dos modos de operación: **tradicional** y **opciones binarias**.

![Demo de la Aplicación](https://via.placeholder.com/800x450.png/282c34/FFFFFF?text=Captura+de+panta+de+la+aplicación)
*(Aquí iría una captura de pantalla o un GIF de tu aplicación en funcionamiento)*

## ✨ Características Principales

-   **Modo de Operación Dual**: Cambia entre un simulador de trading tradicional (con Stop Loss/Take Profit) y un simulador de opciones binarias (con expiración fija).
-   **Múltiples Estrategias Predefinidas**: Prueba estrategias clásicas como Momentum, Mean Reversion, cruce de MACD, y más.
-   **Modelo de Machine Learning**: Utiliza un clasificador Random Forest entrenado con indicadores técnicos para predecir la probabilidad de que el precio suba.
-   **Backtesting Completo**: Simula tus estrategias sobre datos históricos y obtén métricas clave como rentabilidad total, porcentaje de aciertos, factor de beneficio, etc.
-   **Visualización Interactiva**: Gráficos de velas, indicadores técnicos y señales de compra/venta generados con Plotly para un análisis detallado.
-   **Configuración Flexible**: Ajusta todos los parámetros desde la barra lateral (activo, timeframe, parámetros de estrategias, etc.).

## 🧠 ¿Cómo Funciona?

La aplicación sigue un flujo de trabajo claro y automatizado:

1.  **Configuración**: El usuario selecciona el activo financiero (ej. `AAPL`, `BTC-USD`), el timeframe (ej. `5m`, `1h`) y el período de datos desde la barra lateral.
2.  **Obtención de Datos**: La app utiliza la librería `yfinance` para descargar datos históricos del activo seleccionado.
3.  **Análisis Técnico**: Se calculan automáticamente más de 10 indicadores técnicos (EMAs, RSI, MACD, Bandas de Bollinger, VWAP, ATR, etc.) usando `pandas_ta`.
4.  **Generación de Señales**:
    -   Según la estrategia seleccionada, se generan señales de compra (`CALL`) y venta (`PUT`).
    -   Si se elige la estrategia de *Machine Learning*, se entrena un modelo `RandomForestClassifier` para predecir si el precio será mayor en el futuro (en 5 días para modo tradicional, o en el tiempo de expiración para modo binario).
5.  **Backtesting**:
    -   **Modo Tradicional**: Simula operaciones con entrada, salida, Stop Loss y Take Profit.
    -   **Modo Binarias**: Simula operaciones con un precio de ejecución, una fecha de expiración fija y un resultado de gana/pierde.
6.  **Resultados y Visualización**: La aplicación muestra un resumen de las operaciones simuladas con métricas de rendimiento y un gráfico interactivo donde se pueden ver las señales generadas sobre el precio del activo.

## 🚀 Instalación y Puesta en Marcha

Sigue estos pasos para ejecutar la aplicación en tu máquina local.

### Prerrequisitos

-   Python 3.8 o superior.
-   Pip (gestor de paquetes de Python).

### Pasos

1.  **Clona el repositorio**:
    ```bash
    git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
    cd TU_REPOSITORIO
    ```

2.  **Crea un entorno virtual (recomendado)**:
    ```bash
    python -m venv venv
    ```

3.  **Activa el entorno virtual**:
    -   En Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   En macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Instala las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Si no tienes un archivo `requirements.txt`, crea uno con el contenido que te proporciono más abajo).*

5.  **Ejecuta la aplicación**:
    ```bash
    streamlit run app.py
    ```

La aplicación se abrirá automáticamente en tu navegador web, generalmente en la dirección `http://localhost:8501`.

## 📄 Contenido de `requirements.txt`

Crea un archivo llamado `requirements.txt` y añade las siguientes líneas:

```
streamlit
pandas
yfinance
pandas-ta
plotly
numpy
scikit-learn
```

## 🎲 Uso de la Aplicación

La interfaz se divide en la barra lateral de configuración y el área principal de resultados.

### Barra Lateral

1.  **Parámetros de Configuración**: Define el activo, el timeframe y el período de datos a analizar.
2.  **Modo de Operación**:
    -   **Activar Modo Opciones Binarias**: Marca esta casilla para cambiar el simulador. Aparecerán nuevos parámetros como el tiempo de expiración y el porcentaje de pago.
    -   Si está desactivado, se usarán los parámetros del backtester tradicional (Stop Loss, Take Profit, etc.).
3.  **Estrategia de Trading**: Selecciona la estrategia que quieres probar. La opción `Machine Learning (RF)` entrena un modelo en cada ejecución.
4.  **Parámetros de ML**: Si eliges la estrategia de ML, ajusta el umbral de confianza para generar una señal de compra.

### Área Principal

-   **Resultados del Backtester**: Un panel con las métricas clave de tu simulación (rentabilidad, % de aciertos, etc.).
-   **Gráfico Interactivo**: Un gráfico de 4 paneles que muestra el precio y los indicadores. Las señales de compra (triángulo verde) y venta (triángulo rojo) se marcan directamente en el gráfico de precios.

## ⚠️ Aviso Importante

**Esta herramienta es para fines educativos y de investigación únicamente.**

-   No constituye asesoramiento financiero.
-   Los resultados del backtesting se basan en datos históricos y no garantizan rendimientos futuros.
-   El trading, especialmente el de opciones binarias, conlleva un nivel de riesgo muy elevado y puede no ser adecuado para todos los inversores.
-   Opera con dinero real bajo tu propio riesgo.

## 🛠️ Tecnologías Utilizadas

-   **Streamlit**: Para la creación de la interfaz web interactiva.
-   **Pandas**: Para la manipulación y análisis de datos.
-   **yfinance**: Para la descarga de datos de mercado de Yahoo Finance.
-   **pandas-ta**: Para el cálculo de indicadores de análisis técnico.
-   **Plotly**: Para la generación de gráficos interactivos.
-   **Scikit-learn**: Para el modelo de Machine Learning (RandomForestClassifier).
-   **NumPy**: Para operaciones numéricas eficientes.

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Si tienes alguna idea para mejorar la aplicación, puedes:

1.  Hacer un `fork` del proyecto.
2.  Crear una nueva rama (`git checkout -b feature/nueva-caracteristica`).
3.  Realizar tus cambios y hacer `commit` (`git commit -am 'Añadir nueva característica'`).
4.  Pushear a la rama (`git push origin feature/nueva-caracteristica`).
5.  Abrir un `Pull Request`.

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.


Aquí tienes una lista de símbolos populares y viables para usar con `yfinance`, organizados por categoría. He incluido el formato exacto que necesitas pegar en la aplicación.

---

### 📈 Acciones (Stocks) - Mercado Estadounidense

Estos son los más líquidos y populares del mundo. Ideales para cualquier estrategia.

| Símbolo | Nombre de la Empresa | Sector |
| :--- | :--- | :--- |
| `AAPL` | Apple Inc. | Tecnología |
| `MSFT` | Microsoft Corporation | Tecnología |
| `GOOGL` | Alphabet Inc. (Google) | Tecnología |
| `AMZN` | Amazon.com Inc. | Consumo Discrecional |
| `NVDA` | NVIDIA Corporation | Semiconductores |
| `TSLA` | Tesla Inc. | Automoción |
| `META` | Meta Platforms Inc. (Facebook) | Tecnología |
| `JPM` | JPMorgan Chase & Co. | Finanzas |
| `V` | Visa Inc. | Finanzas |
| `WMT` | Walmart Inc. | Consumo Básico |

---

### 🌍 Índices Bursátiles (Stock Indices)

Los índices reflejan el sentimiento general del mercado. Son excelentes para estrategias a medio y largo plazo. **Importante:** Usan el prefijo `^`.

| Símbolo | Nombre del Índice | País |
| :--- | :--- | :--- |
| `^GSPC` | S&P 500 | EE. UU. |
| `^DJI` | Dow Jones Industrial Average | EE. UU. |
| `^IXIC` | NASDAQ Composite | EE. UU. |
| `^GDAXI` | DAX | Alemania |
| `^FTSE` | FTSE 100 | Reino Unido |
| `^N225` | Nikkei 225 | Japón |

---

### 💱 Divisas (Forex)

El mercado de divisas es el más grande del mundo y funciona 24/5. Perfecto para timeframes cortos (`5m`, `15m`, `1h`). **Importante:** Usan el formato `PAR=X`.

| Símbolo | Par de Divisas | Descripción |
| :--- | :--- | :--- |
| `EURUSD=X` | Euro / Dólar Americano | El par más negociado del mundo. |
| `GBPUSD=X` | Libra Esterlina / Dólar | Conocido como "Cable". |
| `USDJPY=X` | Dólar / Yen Japonés | Muy popular en Asia. |
| `AUDUSD=X` | Dólar Australiano / Dólar | Vinculado a materias primas. |
| `USDCAD=X` | Dólar / Dólar Canadiense | "Loonie". |
| `USDCHF=X` | Dólar / Franco Suizo | Considerado un valor refugio. |

---

### 🛢️ Materias Primas (Commodities)

Activos muy influidos por la oferta, la demanda y factores geopolíticos. **Importante:** Usan el sufijo `=F` para contratos de futuros.

| Símbolo | Materias Primas | Descripción |
| :--- | :--- | :--- |
| `GC=F` | Oro (Ounce) | El valor refugio por excelencia. |
| `SI=F` | Plata (Ounce) | El "hermano pequeño" del oro. |
| `CL=F` | Petróleo Crudo WTI (Barrel) | El petróleo más referenciado. |
| `NG=F` | Gas Natural | Muy volátil, sensible al clima. |

---

### ₿ Criptomonedas (Cryptocurrencies)

Un mercado de alta volatilidad, disponible 24/7. Ideal para traders que buscan movimientos grandes y rápidos. **Importante:** Usan el formato `TICKER-USD`.

| Símbolo | Criptomoneda | Descripción |
| :--- | :--- | :--- |
| `BTC-USD` | Bitcoin / USD | La criptomoneda original y líder. |
| `ETH-USD` | Ethereum / USD | La segunda más grande, con contratos inteligentes. |
| `BNB-USD` | Binance Coin / USD | Token del exchange más grande. |
| `XRP-USD` | Ripple / USD | Enfocada en transferencias bancarias. |
| `SOL-USD` | Solana / USD | Conocida por su velocidad. |
| `ADA-USD` | Cardano / USD | Una plataforma de blockchain robusta. |

---

### ⚠️ Consideraciones Importantes

1.  **Liquidez es Clave**: He elegido estos símbolos porque son los que más volumen tienen. Esto significa que los patrones técnicos tienden a ser más fiables.
2.  **Timeframes y Activos**:
    *   **Forex y Cripto**: Funcionan muy bien en timeframes cortos (`5m`, `15m`, `1h`) por su alta volatilidad y actividad continua.
    *   **Acciones e Índices**: A menudo dan mejores resultados en timeframes más largos (`1h`, `1d`), donde el ruido del mercado a corto plazo es menor.
3.  **Cómo Verificar un Símbolo**: ¿No estás seguro si un símbolo funciona? Puedes probarlo rápidamente con este pequeño código en Python:

