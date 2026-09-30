# Neurona artificial para el riego de plantas

Neurona artificial construida con NumPy que decide si una planta necesita ser regada a partir de dos variables: la **humedad del suelo** y la **temperatura ambiental**. Usa la función de activación **sigmoide** y entrega una salida binaria: `1` = regar, `0` = no regar.

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Descripción del problema

| Variable | Descripción | Escala |
|---|---|---|
| x1 | Humedad del suelo (%) | 0 a 100 |
| x2 | Temperatura ambiental (°C) | 0 a 50 aprox. |
| y | Decisión: 1 regar, 0 no regar | binaria |

La neurona calcula `z = x1·w1 + x2·w2 + b`, aplica la sigmoide para obtener una probabilidad entre 0 y 1 y la convierte en 0 o 1 con un umbral de 0.5. Se entrena con descenso de gradiente sobre el error cuadrático medio usando los 10 casos de la guía.

## Instrucciones de ejecución

Requisitos: tener instalado [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<tu-usuario>/neurona-riego-plantas.git
cd neurona-riego-plantas
uv sync
uv run main.py
```

El programa muestra, en este orden: los datos originales y normalizados, el progreso del entrenamiento base, los pesos y el sesgo aprendidos, la probabilidad de cada caso de entrenamiento, las cinco predicciones con condiciones nuevas, la tabla de experimentos y la comparación de umbrales.

## Normalización

Las entradas se dividen por `escala = np.array([100, 50])`, de modo que ambas quedan cerca del intervalo [0, 1]. `X_normalizado` se usa en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). Los datos nuevos se dividen por **la misma** `escala` antes de predecir.

## Resultados del entrenamiento base (tasa 0.5, 10000 épocas)

| Parámetro | Valor |
|---|---|
| Peso de la humedad (w1) | -19.5032 |
| Peso de la temperatura (w2) | 2.3101 |
| Sesgo (b) | 7.4022 |
| Error final (ECM) | 0.019645 |
| Respuestas correctas | 10/10 |

| Caso | Humedad | Temperatura | Probabilidad | Predicción | Esperado |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.001 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.005 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.018 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.103 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.295 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.728 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.850 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.954 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.994 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.999 | 1 | 1 |

**Signo de los pesos:** el peso de la humedad es negativo, así que cuanto más húmedo está el suelo, menor es la probabilidad de regar. El peso de la temperatura es positivo, así que cuanto más calor hace, mayor es la probabilidad de regar. Además, el peso de la humedad es mucho mayor en magnitud (≈19.5 frente a ≈2.3), lo que indica que la humedad es la variable que más pesa en la decisión.

## Predicciones con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 – No regar |
| 45 % | 34 °C | 0.5490 | 1 – Regar |
| 25 % | 22 °C | 0.9719 | 1 – Regar |
| 50 % | 25 °C | 0.2325 | 0 – No regar |
| 30 % | 40 °C | 0.9677 | 1 – Regar |

El caso 45 % / 34 °C queda muy cerca del límite (0.549): está entre los ejemplos "no regar" de 50 % y "regar" de 40 %, y el calor alto lo empuja apenas por encima de 0.5.

## Experimentos con los parámetros

En todos los experimentos se usa la misma semilla (`default_rng(7)`) para los pesos iniciales, así solo cambia el parámetro indicado.

| Experimento | Épocas | Tasa | Error final | Correctas | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | Bueno y estable |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | Aceptable, todavía mejorable |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | Mejora pequeña frente a la base |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | Correcto pero más lento que la base |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | Rápido y estable |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | Muy rápido, sin oscilaciones en este caso |

## Prueba adicional con el umbral

Sin volver a entrenar, se compararon los umbrales 0.4, 0.5 y 0.6 sobre la neurona base. En los 10 casos de entrenamiento **ninguna** decisión cambia, porque todas las probabilidades están lejos de esa franja (la más cercana es el caso 5 con 0.295 y el caso 6 con 0.728). En las condiciones nuevas solo cambia **45 % / 34 °C** (probabilidad 0.549): con umbral 0.4 y 0.5 se decide regar, y con 0.6 se decide no regar.

El umbral no cambia los pesos porque no participa en el entrenamiento: los pesos y el sesgo se ajustan usando la probabilidad continua de la sigmoide. El umbral se aplica después, solo para convertir esa probabilidad en una decisión. Subirlo hace a la neurona más "conservadora" para regar; bajarlo la hace regar con menos evidencia.

## Análisis

**1. ¿Por qué fue necesario normalizar?** La humedad llega a 100 y la temperatura a unos 50. Sin normalizar, la humedad dominaría la suma ponderada y los gradientes solo por tener números más grandes, no por ser más importante. Además, valores de `z` muy grandes saturan la sigmoide (queda en 0 o 1 y su derivada es casi cero), lo que frena el aprendizaje. Con ambas variables cerca de [0, 1] el entrenamiento es más estable y los pesos se pueden comparar entre sí.

**2. ¿Dónde se usó `X_normalizado` y para qué se conservó `X`?** `X_normalizado` se usó en la suma ponderada `z = X_normalizado @ pesos + sesgo` y en el gradiente `X_normalizado.T @ gradiente_z`. `X` se conservó para mostrar los valores en porcentaje y grados Celsius, que son los que una persona entiende al interpretar los resultados.

**3. ¿Qué ocurrió con solo 100 épocas?** El error quedó alto (0.166) y las probabilidades quedaron todas agrupadas entre 0.29 y 0.69. Aunque acertó los 10 casos, lo hizo por muy poco margen (el caso 5 tuvo 0.49 y los casos 6 y 7 tuvieron 0.52): la neurona apenas empezaba a separar las clases y no tiene seguridad en sus respuestas. El aprendizaje fue insuficiente.

**4. ¿Más épocas siempre mejoraron de forma importante?** No. De 100 a 1000 y de 1000 a 10000 épocas la mejora fue grande, pero de 10000 a 20000 el error solo bajó de 0.0196 a 0.0117 y la cantidad de aciertos siguió igual. Después de cierto punto, cada época extra aporta muy poco y solo aumenta el tiempo de cómputo.

**5. ¿Qué efecto tuvo una tasa demasiado pequeña?** Con 0.01 los pasos fueron tan cortos que, después de 10000 épocas, el error (0.130) quedó parecido al de solo 100 épocas con tasa 0.5. El aprendizaje fue muy lento y las probabilidades quedaron poco separadas.

**6. ¿Qué efecto tuvo una tasa alta o muy alta?** Con 1.0 el resultado fue igual al de 20000 épocas con tasa 0.5 (mismo error 0.011705), porque duplicar la tasa equivale aproximadamente a duplicar el recorrido. Con 2.0 se obtuvo el menor error (0.0066) y el error bajó en todas las épocas, sin oscilar. En este problema no apareció inestabilidad porque hay pocos datos, están normalizados y la sigmoide limita el tamaño de los gradientes. Aun así, una tasa muy alta es riesgosa: en problemas con más datos o sin normalizar puede hacer que el error salte o diverja. También se observa que la probabilidad del caso 45 % / 34 °C se acerca a 0.5 a medida que la neurona se ajusta más: la frontera se vuelve más "dura" y ese punto queda casi sobre ella.

**7. Signo del peso de la humedad.** Es negativo: la humedad reduce la probabilidad de regar. Tiene sentido, un suelo húmedo no necesita agua.

**8. Signo del peso de la temperatura.** Es positivo: la temperatura aumenta la probabilidad de regar, porque el calor seca más rápido el suelo. Su magnitud es pequeña, así que influye sobre todo en casos límite de humedad (como 45 %).

**9. ¿Por qué convertir la probabilidad con un umbral?** La sigmoide entrega un valor continuo (por ejemplo 0.73), pero la acción real es discreta: se riega o no se riega. El umbral traduce la probabilidad en una decisión concreta y permite comparar directamente con las salidas esperadas 0 y 1.

**10. Limitaciones para una planta real.** Solo tiene 10 ejemplos inventados y dos variables. No considera el tipo de planta, el suelo, la luz, la lluvia prevista, la humedad del aire, la estación ni la cantidad de agua que se debe aplicar. Al ser una sola neurona, solo puede trazar una frontera lineal entre regar y no regar, y no puede representar relaciones más complejas. Tampoco se evaluó con datos distintos a los de entrenamiento, así que no sabemos qué tan bien generaliza.

## Estructura del repositorio

```
neurona-riego-plantas/
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```
