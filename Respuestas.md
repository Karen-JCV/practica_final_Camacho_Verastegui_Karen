# Respuestas — Práctica Final: Análisis y Modelado de Datos
---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---
El dataset analizado contiene información sobre el uso del smartphone y posibles indicadores de adicción en un total de 7500 usuarios. Incluye variables numéricas relacionadas con el tiempo de uso, número de notificaciones y aperturas de aplicaciones, así como variables categóricas como género, nivel de estrés o impacto académico.

Tras la exploración inicial, se comprobó que los datos están limpios, sin duplicados y con valores nulos únicamente en la variable `addiction_level` (10.92%), los cuales fueron imputados como `"unknown"` para preservar la información y no alterar resultados con la moda.

Las variables numéricas presentan distribuciones mayoritariamente simétricas y sin presencia de outliers según el método IQR. Para la detección de valores atípicos en éste ejercicio se utilizó el método del rango intercuartílico (IQR), ya que no asume normalidad en los datos y es robusto frente a distribuciones sesgadas. Dado que las variables analizadas representan comportamientos de uso (como aperturas de aplicaciones), es esperable la presencia de usuarios extremos, por lo que este método resulta más adecuado que el Z-score, que se basa en la media y desviación estándar y es sensible a outliers. 

En particular, la variable objetivo `app_opens_per_day` tiene un comportamiento estable 
skewness ≈ -0.01), lo que indica ausencia de sesgo.

El análisis de correlaciones muestra que no existen relaciones lineales fuertes con la variable objetivo (coeficientes cercanos a 0), lo que sugiere que el fenómeno es complejo y posiblemente no lineal. Además, se detectó multicolinealidad entre variables de tiempo de uso, lo que deberá tenerse en cuenta en fases posteriores.

En conjunto, el dataset presenta una estructura adecuada para el modelado, aunque requerirá técnicas más avanzadas para capturar patrones relevantes.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de un conjunto de datos sobre uso de smartphone y adicción digital (Smartphone Usage and Addiction dataset) de la plataforma web Kaggle.  
>  
> La variable objetivo seleccionada es `app_opens_per_day`, que representa el número de veces que un usuario abre aplicaciones diariamente.  
>  
> Tiene sentido aplicar un modelo de regresión porque se trata de una variable numérica continua que mide el nivel de interacción con el dispositivo. Predecir este valor permite modelar el comportamiento de uso del smartphone y analizar qué factores influyen en una mayor o menor actividad.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las principales variables numéricas presentan distribuciones relativamente simétricas, con valores de media y mediana cercanos en la mayoría de los casos.  
>  
> En particular, la variable objetivo `app_opens_per_day` muestra una distribución prácticamente simétrica (skewness ≈ -0.01) y una curtosis negativa, lo que indica una distribución platicúrtica (sin colas pronunciadas).  
>  
> Para la detección de outliers se utilizó el método del rango intercuartílico (IQR), adecuado al no asumir normalidad en los datos. No se detectaron valores atípicos en la variable objetivo (límites: -72.5 y 267.5, fuera del rango real de los datos).  
>  
> Por tanto, no fue necesario eliminar ni transformar valores, manteniendo todos los datos para el análisis.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación absoluta con la variable objetivo `app_opens_per_day` son:
>  
> - `daily_screen_time_hours` → r ≈ 0.024  
> - `work_study_hours` → r ≈ 0.022  
> - `weekend_screen_time` → r ≈ 0.021  
>  
> Estos coeficientes son muy bajos, lo que indica que no existe una relación lineal significativa entre estas variables y la variable objetivo. Esto sugiere que el número de aperturas de aplicaciones no puede explicarse mediante relaciones lineales simples.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, únicamente la variable `addiction_level` presenta valores nulos, con un total de 819 registros, lo que representa aproximadamente un **10.92%** del dataset.  
>  
> En lugar de eliminar estas observaciones o imputarlas con la moda, se decidió asignar la categoría `"unknown"`, ya que se trata de una variable categórica relevante y su eliminación podría implicar pérdida de información.  
>  
> El resto de variables no presentan valores nulos, por lo que no fue necesario aplicar tratamientos adicionales.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Se entrenó un modelo de regresión lineal para predecir la variable objetivo `app_opens_per_day`, aplicando previamente un proceso de limpieza, codificación de variables categóricas mediante One-Hot Encoding y escalado de variables numéricas con `StandardScaler`.

Las variables identificadoras (`transaction_id` y `user_id`) fueron eliminadas por no aportar valor predictivo. El dataset fue dividido en entrenamiento (80%) y test (20%) utilizando `train_test_split` con `random_state=42` para garantizar la reproducibilidad de los resultados.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los resultados obtenidos para el modelo de regresión lineal fueron:
>  
> - MAE: 42.28  
> - RMSE: 48.68  
> - R²: -0.00078  
>  
> El MAE indica que, en promedio, el modelo comete un error de aproximadamente 42 aperturas de aplicaciones al día, lo cual es elevado si se compara con la media de la variable objetivo (≈98).  
>  
> El RMSE es aún mayor, lo que sugiere la existencia de errores de predicción significativos en algunos casos, penalizados por el término cuadrático..  
>  
> El coeficiente R² es negativo (-0.00078), lo que implica que el modelo es peor que una predicción basada en la media. Es decir, la regresión lineal no logra explicar la variabilidad de la variable objetivo.  
>
> Estos resultados son coherentes con el análisis del Ejercicio 1, donde se observó que las correlaciones lineales con la variable objetivo eran prácticamente nulas (|r| < 0.03).
>  
> Por tanto, el modelo presenta un claro underfitting, ya que la regresión lineal no es capaz de capturar la complejidad del fenómeno. Esto sugiere que serían necesarios modelos no lineales o técnicas más avanzadas para mejorar el rendimiento.
>  

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio se ha implementado un modelo de regresión lineal múltiple desde cero utilizando NumPy. A través de la formulación matricial de los Mínimos Cuadrados Ordinarios (OLS), se han estimado los coeficientes del modelo y se ha evaluado su rendimiento sobre un conjunto de datos sintético.

Los resultados obtenidos muestran que el modelo es capaz de aproximar correctamente los coeficientes reales y mantener errores dentro de los rangos esperados, lo que valida la implementación realizada.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula β = (XᵀX)⁻¹ Xᵀy permite calcular los coeficientes óptimos de una regresión lineal minimizando el error cuadrático entre los valores reales y los predichos.  
>
> En esencia, transforma el problema en un sistema de ecuaciones que se resuelve mediante álgebra matricial, encontrando la combinación de coeficientes que mejor ajusta los datos.  
>
> La columna de unos se añade para incorporar el término independiente (intercepto, β₀) en el modelo. Sin esta columna, la recta o hiperplano de regresión estaría obligado a pasar por el origen (0,0), lo cual generalmente no representa correctamente los datos.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.86499486     |
| β₁        | 2.0       | 2.06361770     |
| β₂        | -1.0      | -1.11703839    |
| β₃        | 0.5       | 0.43851694     |

> Los coeficientes ajustados son muy cercanos a los valores reales. Las pequeñas diferencias se deben al ruido gaussiano añadido a los datos, lo que impide recuperar exactamente los parámetros originales.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Los valores obtenidos son:
>
> - MAE  = 1.1665  
> - RMSE = 1.4612  
> - R²   = 0.6897  
>
> El MAE y el RMSE se encuentran dentro del rango esperado indicado en el enunciado.  
>
> El valor de R² es algo inferior al valor de referencia (~0.80), pero sigue siendo razonable y positivo, lo que indica que aproximadamente el 69% de la variabilidad de la variable objetivo es explicada por el modelo.  
>
> Esta diferencia puede deberse al ruido gaussiano presente en el conjunto de datos y a la variabilidad del subconjunto de test.
>
> La cercanía entre los coeficientes reales y los estimados, junto con unos errores bajos (MAE y RMSE), confirma que la implementación del modelo es correcta y consistente con la teoría de mínimos cuadrados.
---

## Ejercicio 4 — Series Temporales

---
En este ejercicio se analiza una serie temporal sintética de 6 años de datos diarios (2018–2023), generada a partir de un modelo aditivo que combina tendencia, estacionalidad, ciclos de largo plazo y ruido gaussiano.

Se ha realizado la descomposición de la serie y el análisis del residuo con el objetivo de identificar sus componentes y evaluar si el ruido se comporta como ruido blanco ideal.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, la serie presenta una tendencia claramente creciente y de tipo aproximadamente lineal.
>  
> Se observa un aumento progresivo del nivel de la serie a lo largo del tiempo, pasando de valores cercanos a 50 al inicio hasta valores superiores a 150 al final del periodo.  
>
> La pendiente es constante, lo que indica una tendencia lineal positiva.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, la serie presenta una estacionalidad clara.  
>
> El periodo es aproximadamente de 365 días, lo que corresponde a un patrón anual. 
> 
> La amplitud del componente estacional es significativa, con variaciones aproximadas de ±15 unidades respecto a la tendencia, lo que indica una fuerte influencia estacional en la serie.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, se observan ciclos de largo plazo con un periodo aproximado de 4 años (punto #3 de la función generar_serie_temporal(semilla=42)).
>
> Estos ciclos no son tan evidentes en la serie original debido a la presencia simultánea de tendencia creciente y estacionalidad anual, pero pueden apreciarse mejor en la componente de tendencia obtenida tras la descomposición.
>
> A diferencia de la tendencia, que es monótona y creciente, los ciclos presentan un comportamiento oscilatorio de baja frecuencia, generando ligeras curvaturas sobre la tendencia lineal.
>
> Esto indica la existencia de fluctuaciones adicionales a largo plazo que no siguen un crecimiento constante, sino que alternan fases de aceleración y desaceleración.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo se aproxima bastante a un ruido ideal.  
> 
> - Media: 0.1271 (cercana a 0)  
> - Desviación típica: 3.2220  
> - Asimetría: -0.0509 (cercana a 0)  
> - Curtosis: -0.0610 (cercana a 0)  
> 
> En el test de normalidad de Jarque-Bera se obtiene un p-valor de 0.5766 (> 0.05), por lo que no se rechaza la hipótesis de normalidad.  
> 
> Además, el test ADF indica que el residuo es estacionario (p-valor ≈ 0).  
> 
> Por tanto, el residuo puede considerarse aproximadamente un ruido blanco gaussiano: centrado en cero, sin estructura aparente y con comportamiento aleatorio.

---