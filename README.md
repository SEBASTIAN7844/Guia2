# Guía 2 — Pipeline Modular de Aprendizaje Supervisado

Repositorio: https://github.com/SEBASTIAN7844/Guia2.git

## Descripción del proyecto

Este proyecto implementa un pipeline modular de **aprendizaje supervisado** en Python (scikit-learn) sobre un dataset sintético de rendimiento estudiantil (`data/student_performance_large.csv`, 1000 registros con 6 variables predictoras: horas de estudio semanal, porcentaje de asistencia, horas de sueño, promedio anterior, actividades extracurriculares y nivel de estrés).

Con esos datos se resuelven dos tareas:

- **Regresión**: predecir `nota_final` mediante `LinearRegression`.
- **Clasificación**: predecir `aprobado` (0/1) mediante `LogisticRegression` y `DecisionTreeClassifier`.

### Estructura del proyecto

```
Guia2-main/
├── main.py                        # Orquesta los 3 experimentos del pipeline
├── src/
│   ├── data_loader.py              # Carga, imputación, escalado y train/test split
│   ├── regression_model.py         # Clase RegresorNotas (LinearRegression)
│   └── classification_model.py     # Clase ClasificadorAprobacion (Logistic/Tree)
├── scripts/
│   └── generate_large_dataset.py   # Genera el dataset sintético de 1000 registros
└── data/
    ├── dataset_supervisado.csv     # Dataset base (pequeño)
    └── student_performance_large.csv  # Dataset masivo usado por defecto
```

### Pipeline (`main.py`)

1. **Tarea 1 — Comparación de estrategias de imputación**: entrena y evalúa regresión y clasificación imputando los valores nulos con `mean` y con `median`.
2. **Tarea 2 — Comparación de tamaño de prueba**: entrena el modelo de regresión con `test_size` de 0.20 y 0.50 para observar el efecto en el desempeño.
3. **Tarea 3 — Comparación de algoritmos de clasificación**: compara `LogisticRegression` contra `DecisionTreeClassifier` con el mismo split.

### Preprocesamiento (`data_loader.py`)

- Separación de variables predictoras (`X`) y variables objetivo (`nota_final` para regresión, `aprobado` para clasificación).
- Imputación de nulos en `X` con `SimpleImputer` (estrategia configurable: `mean` o `median`).
- Escalado de `X` con `StandardScaler`.
- División train/test con `train_test_split` (`random_state=42` para reproducibilidad).

### Cómo ejecutar

```bash
pip install pandas scikit-learn
python main.py
```

---

## Informe Requerido

### 1. ¿Cómo afectó la imputación por mediana comparada con la media en el rendimiento final de los modelos?

Con el dataset usado (`student_performance_large.csv`, 408 valores nulos distribuidos en 4 de las 6 variables), los resultados obtenidos fueron:

| Estrategia | MSE (Regresión) | RMSE | R² | Accuracy (Regresión Logística) |
|---|---|---|---|---|
| Media (`mean`) | 2.2367 | 1.4956 | 0.6849 | 99.50% |
| Mediana (`median`) | 2.2337 | 1.4945 | 0.6853 | 99.50% |

La diferencia entre ambas estrategias fue **mínima**: el R² varió apenas de 0.6849 a 0.6853 y el accuracy de clasificación se mantuvo idéntico (99.50%). Esto se explica porque las variables imputadas (`horas_estudio_semana`, `asistencia_pct`, `horas_sueno`, `promedio_anterior`) se generaron con una distribución uniforme y sin outliers extremos, por lo que la media y la mediana son valores muy cercanos entre sí. En general, la mediana suele ser preferible cuando existen valores atípicos (outliers) que distorsionan la media, ya que es una medida robusta a ellos; con datos simétricos y sin outliers, como en este caso, ambas estrategias producen resultados prácticamente equivalentes.

### 2. ¿Qué modelo obtuvo mejor rendimiento en la tarea de clasificación? ¿El Árbol de Decisión o la Regresión Logística? Explique por qué.

| Modelo | Accuracy | Matriz de confusión |
|---|---|---|
| Regresión Logística | 99.50% | [[0, 1], [0, 199]] |
| Árbol de Decisión | 98.00% | [[1, 0], [4, 195]] |

La **Regresión Logística** obtuvo mejor rendimiento (99.50% vs 98.00%). Esto se debe principalmente a que la variable objetivo `aprobado` se define mediante un umbral lineal simple sobre `nota_final` (`nota_final >= 10.5`), y a su vez `nota_final` es una combinación lineal de las variables predictoras más ruido gaussiano. Como la frontera de decisión real es esencialmente lineal, la Regresión Logística —que modela justamente una frontera lineal (tras aplicar la función sigmoide)— se ajusta de forma casi perfecta al problema. El Árbol de Decisión, en cambio, construye particiones rectangulares del espacio de características paso a paso, lo que introduce un ligero sobreajuste a patrones locales del conjunto de entrenamiento y le cuesta más replicar exactamente una frontera lineal continua, resultando en algunos errores adicionales de clasificación.

### 3. ¿Por qué es crítico escalar las variables de entrada (StandardScaler) antes de entrenar un algoritmo como la Regresión Logística?

Es crítico porque la Regresión Logística estima sus coeficientes mediante un algoritmo de optimización basado en gradiente (descenso de gradiente / solvers iterativos), y ese proceso es sensible a la escala de las variables:

- **Convergencia del optimizador**: si las variables tienen escalas muy distintas (por ejemplo, `asistencia_pct` en el rango 50–100 frente a `nivel_estres` en el rango 1–10), la superficie de la función de costo se vuelve muy alargada en ciertas direcciones, lo que hace que el optimizador converja más lento o incluso falle en converger dentro del número máximo de iteraciones.
- **Coeficientes comparables**: al escalar todas las variables a media 0 y desviación estándar 1 (`StandardScaler`), los coeficientes del modelo quedan en una escala comparable entre sí, lo que permite interpretar cuáles variables tienen mayor peso relativo en la predicción.
- **Regularización equilibrada**: la Regresión Logística de scikit-learn aplica regularización (L2) por defecto, la cual penaliza el tamaño de los coeficientes. Si las variables no están en la misma escala, esa penalización afecta de forma desigual a cada variable, perjudicando injustamente a las que tienen valores numéricos más pequeños.

En cambio, algoritmos basados en particiones como el Árbol de Decisión no requieren escalado, porque sus divisiones se basan en umbrales por variable y son invariantes a transformaciones monótonas de escala.

### 4. Si aumentamos la cantidad de datos en el conjunto de prueba (`test_size=0.5`), ¿qué sucede con el rendimiento del modelo?

| Test Size | MSE | RMSE | R² |
|---|---|---|---|
| 0.20 (200 muestras de prueba) | 2.2367 | 1.4956 | 0.6849 |
| 0.50 (500 muestras de prueba) | 2.4884 | 1.5775 | 0.6687 |

Al aumentar `test_size` de 0.20 a 0.50, el modelo de regresión dispone de **menos datos para entrenar** (500 muestras en lugar de 800), lo que provoca una ligera caída en su rendimiento: el MSE y el RMSE aumentan, y el R² disminuye de 0.6849 a 0.6687. Esto ocurre porque, con menos ejemplos de entrenamiento, el modelo tiene menos información para estimar sus parámetros con precisión y generaliza un poco peor. Como contrapartida, la evaluación sobre un conjunto de prueba más grande (500 muestras) es estadísticamente más confiable y menos sensible a la varianza del muestreo que una evaluación sobre solo 200 muestras. En resumen, existe un **trade-off** entre tener más datos de entrenamiento (mejor ajuste del modelo) y tener más datos de prueba (evaluación más robusta), y `test_size=0.5` favorece la robustez de la evaluación a costa del rendimiento del modelo entrenado.
