import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================

def cargar_datos(ruta):
    """
    Carga el dataset desde un archivo CSV.

    Parameters
    ----------
    ruta : str - Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame - DataFrame con los datos cargados.
    """
    return pd.read_csv(ruta, sep=',', encoding='utf-8')

def preprocesar(df):
    """
    Realiza el preprocesamiento del dataset.

    Operaciones:
    - Elimina variables identificadoras ('transaction_id', 'user_id')
    - Aplica One-Hot Encoding a variables categóricas
    - Separa variables predictoras (X) y variable objetivo (y)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple
        X : pd.DataFrame - Variables predictoras
        y : pd.Series - Variable objetivo ('app_opens_per_day')
    """
    # Eliminar IDs
    df = df.drop(columns=['transaction_id', 'user_id'])
    # One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Definir X e y
    target = 'app_opens_per_day'
    X = df.drop(columns=[target])
    y = df[target]

    return X, y

# =============================================================================
# MODELADO
# =============================================================================

def dividir_datos(X, y):
    """
    Divide los datos en conjuntos de entrenamiento y test.

    Parameters
    ----------
    X : pd.DataFrame - Variables predictoras
    y : pd.Series - Variable objetivo

    Returns
    -------
    tuple - X_train, X_test, y_train, y_test
    """
    # Train/Test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def escalar(X_train, X_test):
    """
    Aplica escalado estándar (StandardScaler) a los datos.

    El scaler se ajusta únicamente con los datos de entrenamiento
    y se aplica tanto a train como a test.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple - X_train_scaled : np.ndarray | X_test_scaled : np.ndarray
    """
    # Escalado
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def entrenar_modelo(X_train, y_train):
    """
    Entrena un modelo de regresión lineal.

    Parameters
    ----------
    X_train : np.ndarray - Datos de entrenamiento (escalados)
    y_train : pd.Series

    Returns
    -------
    LinearRegression - Modelo entrenado.
    """
    # Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluar(y_test, y_pred):
    """
    Calcula métricas de evaluación del modelo.

    Métricas:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coeficiente de determinación)

    Además, guarda los resultados en un archivo de texto.

    Parameters
    ----------
    y_test : pd.Series - Valores reales
    y_pred : np.ndarray - Valores predichos

    Returns
    -------
    tuple - (mae, rmse, r2)
    """
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Guardar métricas
    with open("output/ej2_metricas_regresion.txt", "w") as f:
        f.write("Regresión Lineal — Métricas\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    return mae, rmse, r2

def graficar_residuos(y_test, y_pred):
    """
    Genera un gráfico de residuos del modelo.

    Representa los residuos frente a los valores predichos,
    permitiendo evaluar patrones, heterocedasticidad o sesgos.

    Parameters
    ----------
    y_test : pd.Series
    y_pred : np.ndarray
    """
    residuos = y_test - y_pred

    # Gráfico de residuos
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")
    plt.tight_layout()
    plt.savefig("output/ej2_residuos.png")
    plt.close()
    print("  → output/ej2_residuos.png")

def coeficientes(model, columns):
    """
    Calcula y muestra los coeficientes del modelo de regresión.

    Ordena los coeficientes por importancia (valor absoluto).

    Parameters
    ----------
    model : LinearRegression - Modelo entrenado
    columns : list - Nombres de las variables predictoras

    Returns
    -------
    pd.Series - Coeficientes ordenados por importancia.
    """
    # Importancia de variables (coeficientes)
    coef = pd.Series(model.coef_, index=columns).sort_values(key=abs, ascending=False)
    
    print("=" * 50)
    print("RESULTADOS — Coeficientes")
    print("=" * 50)
    print(f"{coef}\n")
    
    return coef

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Carga del dataset
    df = cargar_datos('data/Smartphone_Usage_And_Addiction.csv')

    # Preprocesamiento
    X, y = preprocesar(df)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    columns = X_train.columns
    X_train, X_test = escalar(X_train, X_test)

    # Modelado
    model = entrenar_modelo(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluación
    evaluar(y_test, y_pred)
    
    # Coeficientes
    coeficientes(model, columns)

    # Visualización
    graficar_residuos(y_test, y_pred)

if __name__ == "__main__":
    main()
