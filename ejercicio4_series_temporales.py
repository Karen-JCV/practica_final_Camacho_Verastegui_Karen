import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 días).
      - Ciclos de largo plazo (periodo ~4 años).
      - Ruido gaussiano.

    Parámetros 
    ----------
    semilla : int — Semilla aleatoria para reproducibilidad (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
#  Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera una visualización de la serie temporal completa y la guarda como imagen.

    El gráfico muestra la evolución de la serie a lo largo del tiempo e incluye:
      - Línea temporal de los valores
      - Título descriptivo
      - Etiquetas en los ejes (fecha y valor)
      - Cuadrícula para facilitar la lectura

    Esta visualización permite identificar de forma preliminar patrones como:
      - Tendencias
      - Estacionalidad
      - Variabilidad o ruido

    Parámetros
    ----------
    serie : pd.Series
        Serie temporal con índice de tipo DatetimeIndex.

    Salida
    ------
      - output/ej4_serie_original.png → Imagen del gráfico de la serie completa.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(serie.index, serie.values, color='steelblue', linewidth=1)

    ax.set_title("Serie Temporal Completa", fontsize=14)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
#  Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone una serie temporal en sus componentes principales utilizando
    un modelo aditivo.

    La descomposición separa la serie en:
      - Tendencia (trend)
      - Estacionalidad (seasonal)
      - Residuo o ruido (resid)
      - Serie observada (observed)

    Se utiliza la función seasonal_decompose de statsmodels con:
      - model='additive'
      - period=365 (estacionalidad anual en datos diarios)

    Además, se genera y guarda automáticamente un gráfico con los cuatro
    componentes resultantes.

    Parámetros
    ----------
    serie : pd.Series
        Serie temporal con índice de tipo DatetimeIndex y frecuencia diaria.

    Retorna
    -------
    resultado : statsmodels.tsa.seasonal.DecomposeResult
        Objeto que contiene los componentes descompuestos accesibles como:
        .trend, .seasonal, .resid y .observed

    Salida
    ------
      - output/ej4_descomposicion.png → Imagen con los 4 subgráficos de la descomposición.        
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    resultado = seasonal_decompose(serie, model='additive', period=365)

    fig = resultado.plot()
    fig.set_size_inches(12, 8)

    plt.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches='tight')
    plt.close()

    return resultado

# =============================================================================
#  Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo de una serie temporal descompuesta para
    evaluar si se comporta como ruido blanco gaussiano.

    El análisis incluye:

    1. Limpieza de datos eliminando valores NaN.
    2. Cálculo de estadísticos descriptivos:
        - Media
        - Desviación típica
        - Asimetría
        - Curtosis
    3. Evaluación de normalidad mediante el test de Jarque-Bera.
    4. Evaluación de estacionariedad mediante el test ADF (Augmented Dickey-Fuller).
    5. Análisis de autocorrelación:
        - Función de autocorrelación (ACF)
        - Función de autocorrelación parcial (PACF)
    6. Visualización de la distribución:
        - Histograma del residuo
        - Superposición de la curva normal teórica ajustada

    Parámetros
    ----------
    residuo : pd.Series
        Componente residual obtenido de la descomposición de la serie temporal.

    Salidas
    -------
      - output/ej4_acf_pacf.png          → ACF y PACF del residuo.
      - output/ej4_histograma_ruido.png  → Histograma + curva normal teórica superpuesta.
      - output/ej4_analisis.txt          → Estadísticos numéricos calculados y resultados de los tests.
    
    Notas
    -----
    Un residuo ideal debería cumplir:
      - Media aproximadamente 0
      - Distribución aproximadamente normal
      - Ausencia de autocorrelación
      - Estacionariedad (p-valor ADF < 0.05)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import jarque_bera, norm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # --------------------------------------------------
    # Limpia el residuo (elimina NaN al inicio/fin)
    # --------------------------------------------------
    residuo_limpio = residuo.dropna()

    # --------------------------------------------------
    # Estadísticos bássicos
    # --------------------------------------------------
    media = residuo_limpio.mean()
    std = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis = residuo_limpio.kurtosis()

    # --------------------------------------------------
    # Test de normalidad (Jarque-Bera)
    # --------------------------------------------------
    jb_stat, jb_p = jarque_bera(residuo_limpio)

    # --------------------------------------------------
    # Test ADF (estacionariedad)
    # --------------------------------------------------
    resultado_adf = adfuller(residuo_limpio)
    adf_stat = resultado_adf[0]
    p_adf = resultado_adf[1]

    # --------------------------------------------------
    # Gráfico ACF y PACF del residuo → output/ej4_acf_pacf.png
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(residuo_limpio, ax=axes[0], lags=50)
    plot_pacf(residuo_limpio, ax=axes[1], lags=50)

    axes[0].set_title("ACF del residuo")
    axes[1].set_title("PACF del residuo")

    plt.tight_layout()
    plt.savefig("output/ej4_acf_pacf.png", dpi=150)
    plt.close()

    # --------------------------------------------------
    # Histograma + curva normal
    # --------------------------------------------------
    plt.figure(figsize=(8, 4))

    # Histograma normalizado
    plt.hist(residuo_limpio, bins=30, density=True, alpha=0.6, color='skyblue')

    # Curva normal teórica
    x = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 100)
    y = norm.pdf(x, loc=media, scale=std)

    plt.plot(x, y, 'r-', lw=2)

    plt.title("Histograma del residuo + Normal teórica")
    plt.xlabel("Valor")
    plt.ylabel("Densidad")

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("output/ej4_histograma_ruido.png", dpi=150)
    plt.close()

    # --------------------------------------------------
    # Guardar resultados en TXT
    # --------------------------------------------------
    with open("output/ej4_analisis.txt", "w") as f:
        f.write("=== ANÁLISIS DEL RESIDUO ===\n\n")

        f.write(f"Media: {media:.4f}\n")
        f.write(f"Desviación típica: {std:.4f}\n")
        f.write(f"Asimetría: {asimetria:.4f}\n")
        f.write(f"Curtosis: {curtosis:.4f}\n\n")

        f.write("=== Test de Normalidad (Jarque-Bera) ===\n")
        f.write(f"Estadístico: {jb_stat:.4f}\n")
        f.write(f"p-valor: {jb_p:.4f}\n")
        f.write("Normal" if jb_p > 0.05 else "No normal")
        f.write("\n\n")

        f.write("=== Test ADF (Estacionariedad) ===\n")
        f.write(f"Estadístico: {adf_stat:.4f}\n")
        f.write(f"p-valor: {p_adf:.4f}\n")
        f.write("Estacionario" if p_adf < 0.05 else "No estacionario")
        f.write("\n")

# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Paso 1: Generar la serie (NO modificar la semilla)
    # ------------------------------------------------------------------
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:      {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # ------------------------------------------------------------------
    # Paso 2: Visualizar la serie completa
    # ------------------------------------------------------------------
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # ------------------------------------------------------------------
    # Paso 3: Descomponer
    # ------------------------------------------------------------------
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # ------------------------------------------------------------------
    # Paso 4: Analizar el residuo
    # ------------------------------------------------------------------
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # ------------------------------------------------------------------
    # Resumen de salidas esperadas
    # ------------------------------------------------------------------
    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  [{estado}] output/{s}")

