🤖 Herramienta de Trading con Machine Learning y Soportes/Resistencias
License: MIT

Una aplicación web interactiva construida con Streamlit para analizar activos financieros, probar estrategias de trading y utilizar un modelo de Machine Learning para predecir movimientos del mercado. La herramienta soporta dos modos de operación (tradicional y opciones binarias) e incluye un sistema automático de detección de soportes y resistencias para identificar puntos de entrada y salida clave.

📸 Demo de la Aplicación
(Aquí iría una captura de pantalla o un GIF de tu aplicación en funcionamiento para mostrarla en acción)

✨ Características Principales
🔄 Modo de Operación Dual: Simula trading tradicional con Stop Loss/Take Profit o un simulador de opciones binarias con expiración fija.
📈 Detección Automática de Soportes y Resistencias: Identifica y visualiza zonas clave de soporte y resistencia en el gráfico para ayudar a identificar puntos de entrada y salida.
🧠 Modelo de Machine Learning: Utiliza un clasificador Random Forest entrenado con indicadores técnicos para predecir la probabilidad de que el precio suba.
📊 Múltiples Estrategias Predefinidas: Prueba estrategias clásicas como Momentum, Mean Reversion, cruce de MACD, y más.
📉 Backtesting Completo: Simula tus estrategias sobre datos históricos y obtén métricas clave como rentabilidad total, porcentaje de aciertos, factor de beneficio, etc.
📈 Visualización Interactiva: Gráficos de velas, indicadores técnicos y señales de compra/venta generados con Plotly para un análisis detallado.
⚙️ Configuración Flexible: Ajusta todos los parámetros desde la barra lateral (activo, timeframe, parámetros de estrategias, etc.).
🔄 Actualización de Datos Casi en Tiempo Real: La aplicación se actualiza automáticamente para mostrar los datos más recientes sin necesidad de recargar la página manualmente.
🧠 ¿Cómo Funciona?
La aplicación sigue un flujo de trabajo claro y automatizado:

Configuración: El usuario selecciona el activo financiero (ej. AAPL, BTC-USD), el timeframe (ej. 5m, 1h) y el período de datos desde la barra lateral.
Obtención de Datos: La app utiliza la librería yfinance para descargar datos históricos del activo seleccionado.
Análisis Técnico: Se calculan automáticamente más de 10 indicadores técnicos (EMAs, RSI, MACD, Bandas de Bollinger, VWAP, ATR, etc.) usando pandas_ta.
Generación de Señales:
Según la estrategia seleccionada, se generan señales de compra (CALL) y venta (PUT).
Si se elige la estrategia de Machine Learning, se entrena un modelo RandomForestClassifier para predecir si el precio será mayor en el futuro.
Backtesting:
Modo Tradicional: Simula operaciones con entrada, salida, Stop Loss y Take Profit.
Modo Binarias: Simula operaciones con un precio de ejecución, una fecha de expiración fija y un resultado de gana/pierde.
Resultados y Visualización: La aplicación muestra un resumen de las operaciones simuladas con métricas de rendimiento y un gráfico interactivo donde se pueden ver las señales y los niveles de soporte/resistencia generados sobre el precio del activo.
🚀 Instalación y Puesta en Marcha
Sigue estos pasos para ejecutar la aplicación en tu máquina local.

Prerrequisitos
Python 3.8 o superior.
Pip (gestor de paquetes de Python).
Pasos
Clona el repositorio:
bash

Line Wrapping

Collapse
Copy
1
2
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
Crea un entorno virtual (recomendado):
bash

Line Wrapping

Collapse
Copy
1
python -m venv venv
Activa el entorno virtual:
En Windows:
bash

Line Wrapping

Collapse
Copy
1
venv\Scripts\activate
En macOS/Linux:
bash

Line Wrapping

Collapse
Copy
1
source venv/bin/activate
Instala las dependencias:
bash

Line Wrapping

Collapse
Copy
1
pip install -r requirements.txt
(Si no tienes un archivo requirements.txt, crea uno con el contenido que se proporciona más abajo).
Ejecuta la aplicación:
bash

Line Wrapping

Collapse
Copy
1
streamlit run app.py
La aplicación se abrirá automáticamente en tu navegador web, generalmente en la dirección http://localhost:8501.

📄 Contenido de requirements.txt
Crea un archivo llamado requirements.txt y añade las siguientes líneas:


Line Wrapping

Collapse
Copy
1
2
3
4
5
6
7
streamlit
pandas
yfinance
pandas-ta
plotly
numpy
scikit-learn
🎲 Uso de la Aplicación
La interfaz se divide en la barra lateral de configuración y el área principal de resultados.

Barra Lateral
Parámetros de Configuración: Define el activo, el timeframe y el período de datos a analizar.
Actualización de Datos: Controla el intervalo de actualización automática o actualiza manualmente.
Modo de Operación:
Activar Modo Opciones Binarias: Marca esta casilla para cambiar el simulador. Aparecerán nuevos parámetros como el tiempo de expiración y el porcentaje de pago.
Si está desactivado, se usarán los parámetros del backtester tradicional (Stop Loss, Take Profit, etc.).
Estrategia de Trading: Selecciona la estrategia que quieres probar. La opción Machine Learning (RF) entrena un modelo en cada ejecución.
Soportes y Resistencias: Ajusta los parámetros para la detección automática de estos niveles.
Área Principal
Resultados del Backtester: Un panel con las métricas clave de tu simulación (rentabilidad, % de aciertos, etc.).
Gráfico Interactivo: Un gráfico de 4 paneles que muestra el precio, los indicadores y los niveles de soporte/resistencia. Las señales de compra (triángulo verde) y venta (triángulo rojo) se marcan directamente en el gráfico de precios.
⚠️ Aviso Importante
Esta herramienta es para fines educativos y de investigación únicamente.

No constituye asesoramiento financiero.
Los resultados del backtesting se basan en datos históricos y no garantizan rendimientos futuros.
El trading, especialmente el de opciones binarias, conlleva un nivel de riesgo muy elevado y puede no ser adecuado para todos los inversores.
Opera con dinero real bajo tu propio riesgo.
📚 Guía de Símbolos de Activos
Aquí tienes una lista de símbolos populares y viables para usar con yfinance, organizados por categoría.

📈 Acciones (Stocks) - Mercado Estadounidense
Símbolo
Nombre de la Empresa
Sector
AAPL	Apple Inc.	Tecnología
MSFT	Microsoft Corporation	Tecnología
GOOGL	Alphabet Inc. (Google)	Tecnología
AMZN	Amazon.com Inc.	Consumo Discrecional
NVDA	NVIDIA Corporation	Semiconductores
TSLA	Tesla Inc.	Automoción
META	Meta Platforms Inc.	Tecnología
JPM	JPMorgan Chase & Co.	Finanzas

💱 Divisas (Forex)
Símbolo
Par de Divisas
Descripción
EURUSD=X	Euro / Dólar Americano	El par más negociado del mundo.
GBPUSD=X	Libra Esterlina / Dólar	Conocido como "Cable".
USDJPY=X	Dólar / Yen Japonés	Muy popular en Asia.
AUDUSD=X	Dólar Australiano / Dólar	Vinculado a materias primas.

₿ Criptomonedas (Cryptocurrencies)
Símbolo
Criptomoneda
Descripción
BTC-USD	Bitcoin / USD	La criptomoneda original y líder.
ETH-USD	Ethereum / USD	La segunda más grande, con contratos inteligentes.
BNB-USD	Binance Coin / USD	Token del exchange más grande.
SOL-USD	Solana / USD	Conocida por su velocidad.

⚠️ Consideraciones sobre los Símbolos
Liquidez es Clave: Los símbolos listados son de alta liquidez, lo que suele hacer que los patrones técnicos sean más fiables.
Timeframes y Activos:
Forex y Cripto: Funcionan muy bien en timeframes cortos (5m, 15m, 1h) por su alta volatilidad.
Acciones e Índices: A menudo dan mejores resultados en timeframes más largos (1h, 1d).
🛠️ Tecnologías Utilizadas
Streamlit: Para la creación de la interfaz web interactiva.
Pandas: Para la manipulación y análisis de datos.
yfinance: Para la descarga de datos de mercado de Yahoo Finance.
pandas-ta: Para el cálculo de indicadores de análisis técnico.
Plotly: Para la generación de gráficos interactivos.
Scikit-learn: Para el modelo de Machine Learning (RandomForestClassifier).
NumPy: Para operaciones numéricas eficientes.
🤝 Contribuir
¡Las contribuciones son bienvenidas! Si tienes alguna idea para mejorar la aplicación, puedes:

Hacer un fork del proyecto.
Crear una nueva rama (git checkout -b feature/nueva-caracteristica).
Realizar tus cambios y hacer commit (git commit -am 'Añadir nueva característica').
Pushear a la rama (git push origin feature/nueva-caracteristica).
Abrir un Pull Request.
📜 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
