import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# A) Resumen estructural
# =============================================================================

# Set de visualización
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 160)

# Carga del archivo CSV
file_path = 'data/Smartphone_Usage_And_Addiction.csv'
df = pd.read_csv(file_path, sep=',', encoding='utf-8')

# -------------------------------------------------------------------------
# Exploración de la información general del dataset
# -------------------------------------------------------------------------

# Forma de filas y columnas
print("=" * 50)
print("N° de Observaciones y N° de Variables")
print("=" * 50)
print(f"Observaciones: {df.shape[0]}")
print(f"Variables: {df.shape[1]}\n")

# Información del dataset y uso de memoria
print("=" * 50)
print("Información del dataset y uso de memoria")
print("=" * 50)
df.info()

# Valores nulos suma
print("\n" + "=" * 50)
print("Valores Nulos:")
print("=" * 50)
print(df.isnull().sum())

# Valores nulos media
print("\n" + "=" * 50)
print("Valores Nulos Media:")
print("=" * 50)
print(df.isnull().mean()*100)

# Filas/Observaciones duplicadas
print("\n" + "=" * 50)
print(f"Filas Duplicadas: {df.duplicated().sum()}")
print("=" * 50)

# -------------------------------------------------------------------------
# Limpieza del dataset
# -------------------------------------------------------------------------

## Memoria antes
mem_antes = df.memory_usage(deep=True).sum()

# Imputación en variable categórica 'addiction_level'
df[['addiction_level']] = df[['addiction_level']].fillna('unknown')

# Modificar tipo de dato de str/object a categoria para el ahorro de memoria.
columnas_categoria = [
    "gender",
    "stress_level",
    "academic_work_impact",
    "addiction_level"
]
df[columnas_categoria] = df[columnas_categoria].astype("category")

## Memoria después
mem_despues = df.memory_usage(deep=True).sum()
print("\n" + "=" * 50)
print('Memoria antes:', mem_antes)
print('Memoria después :', mem_despues)
print('Ahorro de Memoria(%):', round((mem_antes - mem_despues) / mem_antes * 100, 2))

# Muestra del dataset después de la limpieza
print('\nDataset después de la limpieza:', df.shape)
print("=" * 50)

# =============================================================================
# B) Estadísticos descriptivos de variables numéricas
# =============================================================================

# Describir variables numéricas y categóricas
print("\n" + "=" * 50)
print("Descripción de variables:")
print("=" * 50)
print(df.describe())

# Guardar el archivo descriptivo
df.describe().to_csv("output/ej1_descriptivo.csv")

# Guardar archivo descriptivo incluyendo la moda y la varianza
resumen = pd.DataFrame({
    'media': df.mean(numeric_only=True),
    'mediana': df.median(numeric_only=True),
    'moda': df.select_dtypes(include=np.number).mode().iloc[0],
    'std': df.std(numeric_only=True),
    'varianza': df.var(numeric_only=True),
    'min': df.min(numeric_only=True),
    'q1': df.quantile(0.25, numeric_only=True),
    'q3': df.quantile(0.75, numeric_only=True),
    'max': df.max(numeric_only=True)
})
resumen.to_csv("output/ej1_descriptivo2.csv")

# Rango intercuartílico (IQR) de la variable objetivo
q1 = df['app_opens_per_day'].quantile(0.25)
q3 = df['app_opens_per_day'].quantile(0.75)
iqr = q3 - q1
print("\n" + "=" * 50)
print(f"IQR: {iqr}")
print("=" * 50)

# Coeficiente de asimetría (skewness) y curtosis para la variable objetivo.
print("\n" + "=" * 50)
print(f"Skewness para 'app_opens_per_day': {df['app_opens_per_day'].skew()}")
print(f"Curtosis para 'app_opens_per_day': {df['app_opens_per_day'].kurt()}")
print("=" * 50)

# =============================================================================
# C) Distribuciones
# =============================================================================

# -------------------------------------------------------------------------
# Histogramas de todas las variables numéricas
# -------------------------------------------------------------------------
numericas = df.select_dtypes(include=['int64', 'float64']).columns

n = len(numericas)
cols = 3
rows = (n // cols) + 1

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numericas):
    axes[i].hist(df[col], bins=20, color='darkorange', edgecolor='black')
    axes[i].set_title(f"Histograma de {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")

# eliminar ejes vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Histogramas de variables numéricas", fontsize=16)
plt.tight_layout()
plt.savefig("output/ej1_histogramas.png")
plt.close()

# -------------------------------------------------------------------------
# Boxplots por variables categóricas
# -------------------------------------------------------------------------

# variable objetivo
target = "app_opens_per_day"

# variables categóricas
categoricas = df.select_dtypes(include=['category']).columns

# excluir variables IDs
categoricas = [col for col in categoricas if col not in ['transaction_id', 'user_id']]

n = len(categoricas)
cols = 2
rows = (n // cols) + (n % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
axes = axes.flatten()

# estilo visual
sns.set_theme(style="whitegrid")

for i, col in enumerate(categoricas):
    sns.boxplot(
        x=col,
        y=target,
        data=df,
        ax=axes[i],
        color="#FF4C4C",
        showfliers=True,
        flierprops=dict(
            marker='o',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=4,
            alpha=0.7
        )
    )
    axes[i].set_title(f"{target} por {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("App opens per day")
    axes[i].tick_params(axis='x', rotation=30)

# eliminar subplots vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# título general
fig.suptitle("Boxplots de app_opens_per_day por variables categóricas", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/ej1_boxplots.png")
plt.close()

# -------------------------------------------------------------------------
# Detección y tratamiento de outliers. Método IQR
# -------------------------------------------------------------------------
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df['app_opens_per_day'] < lower) | (df['app_opens_per_day'] > upper)]
print("\n" + "=" * 50)
print(f"Outliers Lower: {lower}, Upper: {upper} y Cantidad: {len(outliers)}")
print("=" * 50)

# =============================================================================
# D) Variables categóricas
# =============================================================================

# -------------------------------------------------------------------------
# Frecuencia absoluta y relativa de cada categoría. En la última también se 
# hace el análisis de si alguna categoría domina el dataset (desbalance).
# -------------------------------------------------------------------------

print("\n" + "=" * 50)
for col in categoricas:
    print(f"--- {col} ---")
    print("Frecuencia absoluta:")
    print(df[col].value_counts())
    
    print("\nFrecuencia relativa y Desbalance:")
    print(df[col].value_counts(normalize=True))

# -------------------------------------------------------------------------
# Gráfico de barras o de sectores para cada variable categórica.
# -------------------------------------------------------------------------
n = len(categoricas)
cols = 2
rows = (n // cols) + (n % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
axes = axes.flatten()

sns.set_theme(style="whitegrid")

for i, col in enumerate(categoricas):
    sns.countplot(
        x=col,
        data=df,
        ax=axes[i],
        hue=col,
        palette="viridis",
        legend=False
    )
    axes[i].set_title(f"Frecuencia de {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")
    axes[i].tick_params(axis='x', rotation=30)

# eliminar subplots vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Distribución de variables categóricas", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/ej1_categoricas.png")
plt.close()

# =============================================================================
# E) Correlaciones
# =============================================================================

# solo variables numéricas (sin IDs)
df_num = df.select_dtypes(include=['int64', 'float64']) \
           .drop(columns=['transaction_id', 'user_id'], errors='ignore')

corr = df_num.corr(method='pearson')

# -------------------------------------------------------------------------
# Heatmap
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Matriz de correlación (Pearson)")
plt.tight_layout()
plt.savefig("output/ej1_heatmap_correlacion.png")
plt.close()

# -------------------------------------------------------------------------
# Top 3 variables más correlacionadas con la variable objetivo
# -------------------------------------------------------------------------
target = "app_opens_per_day"

corr_target = corr[target].drop(target)

top3 = corr_target.abs().sort_values(ascending=False).head(3)
print("\n" + "=" * 50)
print("Top 3 variables más correlacionadas con target:")
print("=" * 50)
print(top3)

# -------------------------------------------------------------------------
# Detección de posible multicolinealidad entre predictoras (pares con |r| > 0,9)
# -------------------------------------------------------------------------

# matriz sin diagonal
corr_matrix = corr.abs()

# pares con alta correlación
high_corr = []

for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

print("\n" + "=" * 50)
print(f"Multicolinealidad: {high_corr}\n")
