import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresión Lineal Múltiple usando OLS y devuelve
    las predicciones sobre el conjunto de test.

    La solución analítica o formulación matricial es:
        β = (XᵀX)⁻¹ Xᵀy

    En esta implementación se utiliza np.linalg.lstsq por ser más
    estable numéricamente que la inversión directa.

    Parámetros
    ----------
    X_train : np.ndarray de forma (n_train, p)
        Matriz de features o variables predictoras de entrenamiento. Cada fila 
        es una observación y cada columna es una variable predictora.
    y_train : np.ndarray de forma (n_train,)
        Vector de valores objetivo de entrenamiento.
    X_test : np.ndarray de forma (n_test, p)
        Matriz de features o variables predictoras del conjunto de test.
        
    Retorna
    -------
    coefs : np.ndarray de forma (p+1,)
        Vector de coeficientes ajustados o estimados [β₀, β₁, ..., βₚ].
        donde β₀ es el intercepto (término independiente).
    y_pred : np.ndarray de forma (n_test,)
        Predicciones del modelo sobre X_test.
    """

    # Paso 1 — Añadir columna de unos (intercepto β₀)
    n_train = X_train.shape[0]
    ones_train = np.ones((n_train, 1))
    X_train_b = np.hstack([ones_train, X_train])

    # Paso 2 — Calcular los coeficientes β con la fórmula OLS
    # β = (XᵀX)⁻¹ Xᵀy
    coefs = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]

    # (Alternativa menos estable):
    # coefs = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

    # Paso 3 — Añadir columna de unos a X_test de la misma forma
    n_test = X_test.shape[0]
    ones_test = np.ones((n_test, 1))
    X_test_b = np.hstack([ones_test, X_test])

    # Paso 4 — Calcular predicciones ŷ = X_test_b · β
    y_pred = X_test_b @ coefs

    return coefs, y_pred


# =============================================================================
# FUNCIONES DE MÉTRICAS
# =============================================================================

def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE), que mide el error medio
    absoluto entre valores reales y predichos.

    Fórmula:
        MAE = (1/n) * Σ |y_real - y_pred|

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales.
    y_pred : np.ndarray — Valores predichos por el modelo.

    Retorna
    -------
    float — Valor del MAE.
    """
    return np.mean(np.abs(y_real - y_pred))


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE), que penaliza más
    los errores grandes al elevarlos al cuadrado.

    Fórmula:
        RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales.
    y_pred : np.ndarray — Valores predichos por el modelo.

    Retorna
    -------
    float — Valor del RMSE.
    """
    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R², que indica
    la proporción de la varianza explicada por el modelo.

    Fórmula:
        R² = 1 - SS_res / SS_tot
    donde:
        SS_res = Σ (y_real - y_pred)²
        SS_tot = Σ (y_real - media(y_real))²

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales.
    y_pred : np.ndarray — Valores predichos por el modelo.

    Retorna
    -------
    float — Valor de R² (entre -∞ y 1; cuanto más cercano a 1, mejor, 
    es decir, puede ser negativo, máximo 1).
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)

    return 1 - (ss_res / ss_tot)


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN
# =============================================================================

def graficar_real_vs_predicho(y_real, y_pred, ruta_salida="output/ej3_predicciones.png"):
    """
    Genera un gráfico de dispersión (scatter plot) comparando los valores
    reales frente a los valores predichos por el modelo.

    Incluye una línea de referencia (todos los puntos sobre la diagonal) y = x, 
    que representa un modelo perfecto. La dispersión alrededor de esa línea 
    representa el error del modelo.
    
    Parámetros
    ----------
    y_real : np.ndarray — Valores reales del test set.
    y_pred : np.ndarray — Predicciones del modelo.
    ruta_salida : str   — Ruta donde se guardará la imagen generada.

    Retorna
    -------
    None — Guarda la figura en disco.
    """
    plt.figure(figsize=(6, 6))

    # Scatter
    plt.scatter(y_real, y_pred, alpha=0.6)

    # Línea ideal o perfecta y = x
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Etiquetas
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("Comparación Real vs Predicho")

    # Guardar gráfico
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sintéticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    # Generamos features aleatorias
    X = rng.standard_normal((n_muestras, n_features))

    # Coeficientes "reales" conocidos: β₀=5, β₁=2, β₂=-1, β₃=0.5
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])

    # Variable objetivo con ruido gaussiano (σ=1.5)
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # -------------------------------------------------------------------------
    # Split Train / Test (80% / 20%) — sin mezclar aleatoriamente
    # -------------------------------------------------------------------------
    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Mostrar resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print(f"\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # -------------------------------------------------------------------------
    # RESULTADO DE REFERENCIA DEL PROFESOR
    # Con SEMILLA=42, los resultados esperados aproximados son:
    #   Coefs ajustados ≈ [5.0, 2.0, -1.0, 0.5]  (cercanos a los reales)
    #   MAE  ≈ 1.20  (±0.20 según implementación)
    #   RMSE ≈ 1.50  (±0.20 según implementación)
    #   R²   ≈ 0.80  (±0.05 según implementación)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Guardar salidas
    # -------------------------------------------------------------------------

    # Fichero de coeficientes
    with open("output/ej3_coeficientes.txt", "w") as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β₀)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        f.write("=" * 50 + "\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")

    # Fichero de métricas
    with open("output/ej3_metricas.txt", "w") as f:
        f.write("Regresión Lineal Múltiple — Métricas de evaluación\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    # Gráfico
    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en la carpeta output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
