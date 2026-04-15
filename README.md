# 📊 Smartphone Usage & Addiction Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Descripción del proyecto

Este proyecto tiene como objetivo realizar un análisis completo del comportamiento de uso del smartphone y su relación con posibles indicadores de adicción digital.

A lo largo del proyecto se aplican técnicas de:
- Análisis estadístico descriptivo (EDA)
- Modelado predictivo
- Implementación de algoritmos desde cero
- Análisis de series temporales

El trabajo se estructura en 4 ejercicios que cubren todo el pipeline de análisis de datos.

---

## 📂 Dataset

El dataset contiene información de **7500 usuarios** e incluye variables relacionadas con:

- Tiempo de uso del smartphone
- Redes sociales, gaming y estudio/trabajo
- Notificaciones y aperturas de apps
- Nivel de estrés e impacto académico
- Indicadores de adicción

### 🔗 Fuente

Dataset obtenido de Kaggle:

https://www.kaggle.com/datasets/zahranusratt/smartphone-usage-and-addiction-analysis-dataset/data

Los datos han sido utilizados únicamente con fines educativos.

### 🎯 Variable objetivo

app_opens_per_day

Variable numérica continua que representa el número de aperturas de aplicaciones diarias.

---

# 🧪 Ejercicio 1 — Análisis Estadístico Descriptivo

### 🔍 Objetivo
Comprender la estructura y calidad de los datos.

### 📊 Resultados clave

- Dataset limpio, sin duplicados
- Valores nulos únicamente en `addiction_level` (**10.92%**), tratados como `"unknown"`
- Distribuciones mayoritariamente **simétricas**
- No se detectan outliers en la variable objetivo (método IQR)
- Variables categóricas equilibradas

### ❗ Hallazgos importantes

- No existen correlaciones lineales significativas con la variable objetivo  
  (|r| < 0.03)
- Se detecta **multicolinealidad fuerte** (r > 0.9) entre:
  - `daily_screen_time_hours`
  - `weekend_screen_time`

### 🧠 Conclusión

El comportamiento del uso del smartphone es **complejo y no lineal**, lo que anticipa dificultades para modelos simples.

---

# 🤖 Ejercicio 2 — Regresión Lineal (Scikit-Learn)

### ⚙️ Preprocesamiento

- Eliminación de IDs
- One-Hot Encoding
- Escalado con `StandardScaler`
- Split 80/20

### 📊 Resultados

| Métrica | Valor |
|--------|------|
| MAE    | 42.28 |
| RMSE   | 48.68 |
| R²     | -0.000804 |

### 📉 Interpretación

- Error medio ≈ **42 aperturas/día** (~43% de la media)
- R² negativo → peor que predecir la media
- Errores elevados y dispersión alta

👉 El modelo presenta **underfitting severo**

### 🔍 Conexión con EDA

Estos resultados confirman el Ejercicio 1:
- Ausencia de relaciones lineales
- Variables poco predictivas individualmente

### ⚠️ Conclusión

La regresión lineal **no es adecuada** para este problema.

---

# 🧠 Ejercicio 3 — Regresión Lineal desde cero (NumPy)

### 🎯 Objetivo
Implementar OLS manualmente para comprender la base matemática:

\[
\beta = (X^T X)^{-1} X^T y
\]

### 📊 Resultados

**Coeficientes:**
- Reales: [5.0, 2.0, -1.0, 0.5]  
- Ajustados: [4.8649, 2.0636, -1.1170, 0.4385]

**Métricas:**
- MAE  = 1.1665  
- RMSE = 1.4612  
- R²   = 0.6897  

### 📈 Interpretación

- El modelo recupera correctamente los coeficientes
- R² ≈ 0.69 → explica ~69% de la variabilidad
- Diferencias debidas al ruido gaussiano

### ✅ Conclusión

La implementación es correcta y consistente con la teoría.

---

# 📈 Ejercicio 4 — Análisis de Series Temporales

### 🔍 Descripción

Serie sintética (2018–2023) con:
- Tendencia lineal creciente
- Estacionalidad anual
- Ciclos de largo plazo
- Ruido gaussiano

---

## ⚙️ Metodología

### 1. Visualización
Identificación de patrones globales

### 2. Descomposición aditiva
\[
Serie = Tendencia + Estacionalidad + Residuo
\]

### 3. Análisis del residuo

- Media: **0.1271**
- Std: **3.2220**
- Asimetría ≈ 0
- Curtosis ≈ 0

**Tests:**
- Jarque-Bera → p = 0.5766 (normal)
- ADF → p ≈ 0 (estacionario)

---

## 📊 Resultados

### ✔ Tendencia
- Lineal y creciente (≈ 50 → 150)

### ✔ Estacionalidad
- Periodo ≈ 365 días
- Amplitud ≈ ±15

### ✔ Ciclos
- Oscilaciones suaves de baja frecuencia
- Observables en la tendencia descompuesta

### ✔ Ruido
- Aproximadamente gaussiano
- Sin autocorrelación
- Estacionario

---

## 🧠 Conclusión

La descomposición captura correctamente toda la estructura de la serie.

El residuo cumple propiedades de **ruido blanco**, lo que valida el modelo aditivo.

---

# 📁 Estructura del proyecto
```
.
├── data
│   └── Smartphone_Usage_And_Addiction.csv
├── ejercicio1_descriptivo.py
├── ejercicio2_inferencia.py
├── ejercicio3_regresion_multiple.py
├── ejercicio4_series_temporales.py
├── output
│   ├── ej1_boxplots.png
│   ├── ej1_categoricas.png
│   ├── ej1_descriptivo.csv
│   ├── ej1_descriptivo2.csv
│   ├── ej1_heatmap_correlacion.png
│   ├── ej1_histogramas.png
│   ├── ej2_metricas_regresion.txt
│   ├── ej2_residuos.png
│   ├── ej3_coeficientes.txt
│   ├── ej3_metricas.txt
│   ├── ej3_predicciones.png
│   ├── ej4_acf_pacf.png
│   ├── ej4_analisis.txt
│   ├── ej4_descomposicion.png
│   ├── ej4_histograma_ruido.png
│   └── ej4_serie_original.png
├── readme.md
├── requirements.txt
└── Respuestas.md
```

---

# 🛠️ Tecnologías

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-Learn
- Statsmodels
- SciPy

---

# ⭐ Key Takeaway

> No todos los problemas pueden resolverse con modelos lineales.  
> Entender los datos es más importante que aplicar algoritmos complejos sin análisis previo.

---

# 👨‍💻 Autor

Proyecto desarrollado como práctica final de análisis y modelado de datos.