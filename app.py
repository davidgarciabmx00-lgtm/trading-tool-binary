# --- NUEVO: Función para calcular Soportes y Resistencias (Versión Corregida y Robusta) ---
def calculate_support_resistance(df, window=5, threshold_pct=0.5):
    """
    Calcula los niveles de soporte y resistencia usando un enfoque de fractales.
    Un fractal es un pico o valle local.
    Esta versión es más robusta y maneja casos donde no se encuentran fractales.
    """
    highs = df['High']
    lows = df['Low']
    
    fractal_highs_idx = []
    fractal_lows_idx = []
    
    # Un pico fractal es más alto que los picos a su alrededor
    for i in range(window // 2, len(highs) - window // 2):
        is_high_fractal = True
        is_low_fractal = True
        current_high = highs.iloc[i]
        current_low = lows.iloc[i]
        
        for j in range(i - window // 2, i + window // 2 + 1):
            if j == i:
                continue
            if highs.iloc[j] >= current_high:
                is_high_fractal = False
            if lows.iloc[j] <= current_low:
                is_low_fractal = False
        
        if is_high_fractal:
            fractal_highs_idx.append(i)
        if is_low_fractal:
            fractal_lows_idx.append(i)

    # Obtener los valores de precio de los fractales
    fractal_highs = df['High'].iloc[fractal_highs_idx].values
    fractal_lows = df['Low'].iloc[fractal_lows_idx].values
    
    # --- CORRECCIÓN: Función de agrupación más robusta ---
    def cluster_levels(levels):
        # Convertir a array de numpy para asegurar consistencia
        levels_array = np.array(levels)
        
        # Comprobación explícita y segura para un array vacío
        if levels_array.size == 0:
            return []
            
        # Ordenar el array
        levels_array.sort()
        
        clustered = []
        current_cluster = [levels_array[0]]
        
        for i in range(1, len(levels_array)):
            cluster_mean = np.mean(current_cluster)
            # Evitar división por cero si la media del clúster es 0
            if cluster_mean == 0:
                 if abs(levels_array[i] - cluster_mean) < (threshold_pct / 100):
                     current_cluster.append(levels_array[i])
                 else:
                     clustered.append(cluster_mean)
                     current_cluster = [levels_array[i]]
            else:
                 if abs(levels_array[i] - cluster_mean) / cluster_mean * 100 < threshold_pct:
                     current_cluster.append(levels_array[i])
                 else:
                     clustered.append(cluster_mean)
                     current_cluster = [levels_array[i]]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
            
        return clustered

    resistance_levels = cluster_levels(fractal_highs)
    support_levels = cluster_levels(fractal_lows)
    
    return support_levels, resistance_levels
