import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# Carga del archivo CSV
# =============================================================================

# Ruta del archivo .csv y carga en DataFrame de pandas.
file_path = 'data/Smartphone_Usage_And_Addiction.csv'
df = pd.read_csv(file_path, sep=',', encoding='utf-8')

# =============================================================================
# Preprocesamiento
# =============================================================================

# Eliminar IDs
df = df.drop(columns=['transaction_id', 'user_id'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# =============================================================================
# Modelo A — Regresión Lineal (LinearRegression)
# =============================================================================

# Definir X e y
target = 'app_opens_per_day'
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)

# Métricas
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

# Residuos
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

# Importancia de variables (coeficientes)
coeficientes = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("=" * 50)
print("RESULTADOS — Coeficientes")
print("=" * 50)
print(f"{coeficientes}\n")