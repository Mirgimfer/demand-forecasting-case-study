# Previsión de demanda con series temporales 

Sistema de predicción de demanda para un catálogo de productos agroquímicos,
desarrollado como proyecto de ciencia de datos aplicada. Mi objetivo era
pronosticar la demanda mensual de cientos de series de producto y evaluar
qué técnicas de modelado funcionan mejor según el patrón de comportamiento
de cada serie.

 
> **Nota de confidencialidad.** Este proyecto se realizó en un entorno empresarial.
> Por acuerdo de confidencialidad **no se incluyen datos, código ni información del cliente**.
> Este documento describe únicamente el enfoque metodológico y las decisiones de diseño,
> a modo de caso de estudio.

---

## Contexto

La demanda de productos agroquímicos es mayoritariamente **intermitente**:
muchos meses sin ventas intercalados con picos esporádicos, un régimen difícil
de pronosticar con métodos clásicos pensados para demanda continua. Abordé este
reto con una comparativa sistemática de modelos estadísticos, de machine
learning y de deep learning, organizada por grupos de series de comportamiento
homogéneo.

Desarrollé el trabajo en **dos análisis sucesivos**. En el primero establecí el
planteamiento inicial (clasificación por matriz de Syntetos y modelos entrenados
por clase); en el segundo lo rediseñé a partir de lo aprendido, sustituyendo la
clasificación por *clustering* KMeans y adoptando un entrenamiento global, para
obtener una comparación más limpia. Ambos se documentan abajo.

---

## Primer análisis: enfoque por clasificación de Syntetos

En el planteamiento inicial me apoyé en la clasificación clásica de demanda
intermitente y en un modelado segmentado por clase.

- **Datos:** trabajé con **dos representaciones**, formato ancho (fechas en
  filas, un producto por columna) y formato largo.
- **Clasificación:** clasifiqué las series con la **matriz de Syntetos-Boylan**
  en cuatro clases: *Regular*, *Errática*, *Intermitente* y *Lumpy*.
- **Modelado segmentado:** entrené los modelos **por separado dentro de cada
  clase**.
- **Selección automática de modelo:** para cada clase elegía el modelo de mejor
  rendimiento según las métricas de evaluación.
- **Análisis exploratorio de País:** construí una **matriz de correlación** para
  identificar los países y productos más correlacionados entre sí.

### Limitación detectada

Al intentar responder la pregunta central —**¿aporta valor añadir la variable
País?**— me encontré con un problema de diseño: como había entrenado los modelos
segmentados por clase, al introducir la granularidad de País las series **no
caían necesariamente en la misma clase**, de modo que los dos escenarios (con
País y sin País) no eran directamente comparables. El entrenamiento segmentado
contaminaba la comparación.

Además, al explorar una segmentación alternativa mediante **KMeans**, observé
que los grupos quedaban **mejor definidos y más separados** que con la matriz de
Syntetos. Ambas observaciones me llevaron a rediseñar por completo el estudio en
un segundo análisis.

---

## Segundo análisis: enfoque por clustering y entrenamiento global

Este es el análisis principal del repositorio. Rediseñé el anterior para
eliminar la limitación detectada y obtener una comparativa válida de la
variable País.

### 1. Preparación de datos

- Carga y limpieza de los registros de demanda.
- Agregación a granularidad mensual.
- Relleno explícito de huecos con ceros, para que la intermitencia quede
  representada de forma íntegra en la serie.
- Adopté el formato **largo** (*tidy*: `unique_id`, `ds`, `y`) de principio a
  fin, que es el que exige el ecosistema Nixtla y el natural para modelos
  globales. La granularidad de modelado se controla con lo que se codifica en el
  `unique_id`, sin cambiar el resto del *pipeline*.

### 2. Segmentación de series por comportamiento

Caractericé cada serie con un conjunto de *features* de comportamiento
(tendencia, estacionalidad, volumen, recencia, grado de intermitencia) y las
agrupé en clústeres mediante **KMeans**.

Los clústeres resultantes distinguen, entre otros, patrones de demanda
intermitente activa, demanda regular de alto volumen, series esporádicas en
declive y series en caída pronunciada.

### 3. Modelado y evaluación

Entrené y comparé una batería de modelos, agrupados en tres familias:

- **Estadísticos:** *Naive* (varios), SBA (*Syntetos-Boylan Approximation*,
  derivado de Croston para demanda intermitente) y Prophet.
- **Machine learning:** XGBoost con ingeniería de *features* temporales (*lags*,
  ventanas móviles, variables de calendario), con especial cuidado en evitar
  fugas de información (*leakage*).
- **Deep learning:** GRU, DeepAR (probabilístico) y TFT (*Temporal Fusion
  Transformer*), vía NeuralForecast.

**Esquema de validación:** origen móvil con horizontes de 3 y 6 meses.

**Métrica principal:** MASE (agregado por mediana y por clúster). La escogí por
ser escala-libre y robusta a la presencia masiva de ceros, donde métricas de
porcentaje como MAPE o sMAPE se rompen.

### 4. Análisis probabilístico

Para DeepAR, que produce una distribución completa por mes (no solo un punto),
evalué la calidad de la incertidumbre con métricas específicas:

- **Coverage** (cobertura empírica de los intervalos de predicción).
- **CRPS** (equivalente probabilístico del MAE).
- **Interval Score / Winkler** (penaliza intervalos demasiado anchos o que no
  atrapan el valor real).

El interés de estas métricas es práctico: los cuantiles altos de la
distribución alimentan directamente el dimensionamiento del **stock de
seguridad**.

---

## Qué cambió entre los dos análisis, y por qué

La tabla resume las dos decisiones de diseño que distinguen ambos enfoques y la
razón por la que hice cada cambio.

| Aspecto | Primer análisis | Segundo análisis |
|---|---|---|
| Segmentación de series | Matriz de Syntetos (4 clases) | KMeans sobre *features* de comportamiento |
| Representación de datos | Formato ancho **y** largo | Formato largo único |
| Entrenamiento | Segmentado por clase | Global (todas las series a la vez) |
| Uso de la segmentación | Entrenar **y** evaluar | Solo agrupar métricas en evaluación |

**KMeans en lugar de Syntetos.** La matriz de Syntetos parte el espacio con
umbrales fijos y dejaba clases poco pobladas o con fronteras poco nítidas. Con
KMeans los grupos resultaron **mejor separados y más diferenciados entre sí**,
lo que da grupos más coherentes internamente y hace más informativa la
comparación de modelos dentro de cada uno.

**Entrenamiento global en lugar de segmentado.** Es la decisión que hace válida
la comparación de la variable País. Al entrenar de forma global, ambos brazos
del experimento (con País y sin País) comparten el mismo esquema de
entrenamiento y solo cambia la variable bajo estudio, de modo que cualquier
diferencia observada es atribuible a esa variable y no al reparto de los datos.
Reservé la segmentación por clúster para **agrupar las métricas** en la fase de
evaluación, no para entrenar.

---

## Estructura del proyecto

```
src/demanda/
├── config.py            # Configuración central (rutas, constantes, parámetros)
├── data_loading.py      # Carga y limpieza de datos crudos
├── features.py          # Ingeniería de features y clustering
├── evaluation.py        # Split temporal, filtrado y orquestación de predicciones
├── metrics.py           # Métricas de punto (MASE, MAE, etc.)
├── probabilistico.py    # Métricas probabilísticas (coverage, CRPS, Winkler)
├── storage.py           # Persistencia de predicciones (parquet) y tablas
└── models/
    ├── baselines.py     # Naive y SBA
    └── ...              # XGBoost, modelos DL

notebooks/               # Análisis exploratorio y comparativas
outputs/                 # Predicciones y tablas de resultados (no versionadas)
data/                    # Datos crudos (NO incluidos por confidencialidad)
```

---

## Flujo de trabajo

Entreno los modelos de deep learning en **Google Colab** (GPU) y el resto del
análisis en local. Mantuve la frontera entre ambos entornos al mínimo: solo se
transfieren los datos de entrenamiento (hacia Colab) y las predicciones
resultantes (de vuelta a local), en formato **parquet**. La evaluación se hace
íntegramente en local con un pipeline único, de modo que todos los modelos
—estadísticos, ML y DL— se comparan de forma homogénea.

---

## Stack técnico

- **Python** · pandas · numpy
- **scikit-learn** (KMeans, *features* de comportamiento)
- **XGBoost**
- **Nixtla NeuralForecast** (GRU, DeepAR, TFT) · **MLForecast** (XGBoost global)
- **Prophet**
- **pyarrow** (persistencia en parquet)
