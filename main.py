import numpy as np

print("=" * 65)
print("  NEURONA ARTIFICIAL PARA EL RIEGO DE PLANTAS")
print("=" * 65)
print("Entradas: x1 = humedad del suelo (%), x2 = temperatura (°C)")
print("Salida:   1 = regar, 0 = no regar")

# ---------------------------------------------------------------
# 1. Datos de entrenamiento
# ---------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

print("\nDatos originales (humedad, temperatura):")
print(X)

# ---------------------------------------------------------------
# 2. Normalización de las entradas
# ---------------------------------------------------------------
# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
#
# Dividimos cada columna por su valor máximo esperado
# para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


# ---------------------------------------------------------------
# 3. Función de activación
# ---------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------
# 4. Entrenamiento de la neurona
# ---------------------------------------------------------------
def entrenar(tasa_aprendizaje, epocas, mostrar_progreso=False):
    """Entrena la neurona y devuelve pesos, sesgo y error final."""
    # Misma semilla en todos los experimentos para que solo cambie
    # el parámetro que estamos probando.
    rng = np.random.default_rng(7)
    pesos = rng.normal(size=(2, 1))  # [peso humedad, peso temperatura]
    sesgo = 0.0

    intervalo = max(epocas // 10, 1)

    for epoca in range(epocas):
        # Suma ponderada con los datos NORMALIZADOS
        z = X_normalizado @ pesos + sesgo
        probabilidades = sigmoide(z)

        # Error y gradientes (error cuadrático medio)
        error = probabilidades - y
        gradiente_z = 2 * error * probabilidades * (1 - probabilidades) / len(X_normalizado)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        # Actualización de pesos y sesgo
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar_progreso and epoca % intervalo == 0:
            ecm = np.mean(error ** 2)
            print(f"Época {epoca:6d} | error cuadrático medio: {ecm:.6f}")

    probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
    error_final = np.mean((probabilidades - y) ** 2)
    return pesos, sesgo, error_final


def predecir(humedad, temperatura, pesos, sesgo, umbral=0.5):
    """Normaliza un dato nuevo con la MISMA escala y predice."""
    dato = np.array([[humedad, temperatura]], dtype=float) / escala
    probabilidad = sigmoide(dato @ pesos + sesgo).item()
    decision = int(probabilidad >= umbral)
    return probabilidad, decision


def texto_decision(decision):
    return "REGAR" if decision == 1 else "NO REGAR"


# ---------------------------------------------------------------
# 5. Prueba base: tasa 0.5 y 10000 épocas
# ---------------------------------------------------------------
tasa_aprendizaje = 0.5
epocas = 10000

print("\n" + "=" * 65)
print(f"ENTRENAMIENTO BASE (tasa = {tasa_aprendizaje}, épocas = {epocas})")
print("=" * 65)
pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar_progreso=True)

print(f"\nPeso de la humedad:     {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo:                  {sesgo:.4f}")
print(f"Error final (ECM):      {error_final:.6f}")

print("\nSignificado del signo de los pesos:")
print(" - Peso de la humedad NEGATIVO: a más humedad, menor probabilidad de regar.")
print(" - Peso de la temperatura POSITIVO: a más temperatura, mayor probabilidad de regar.")

# Probabilidad para cada caso de entrenamiento
probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

print("\nResultados sobre los datos de entrenamiento:")
print(f"{'Caso':>4} | {'Humedad':>7} | {'Temp':>5} | {'Prob.':>6} | {'Pred':>4} | {'Esperado':>8}")
print("-" * 52)
for i, (entrada, prob, resp, esperado) in enumerate(
    zip(X, probabilidades.ravel(), respuestas.ravel(), y.ravel().astype(int)), start=1
):
    marca = "OK" if resp == esperado else "X"
    print(f"{i:>4} | {entrada[0]:>6.0f}% | {entrada[1]:>4.0f}° | {prob:>6.3f} | {resp:>4} | {esperado:>8} {marca}")

correctas = int(np.sum(respuestas == y.astype(int)))
print(f"\nRespuestas correctas: {correctas}/10")

# ---------------------------------------------------------------
# 6. Pruebas con condiciones nuevas
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("PRUEBAS CON CONDICIONES NUEVAS (normalizadas con la misma escala)")
print("=" * 65)
condiciones_nuevas = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]

print(f"{'Humedad':>7} | {'Temp':>5} | {'Probabilidad':>12} | Decisión")
print("-" * 45)
for humedad, temperatura in condiciones_nuevas:
    prob, dec = predecir(humedad, temperatura, pesos, sesgo)
    print(f"{humedad:>6}% | {temperatura:>4}° | {prob:>12.4f} | {dec} ({texto_decision(dec)})")

# ---------------------------------------------------------------
# 7. Experimentos con los parámetros
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("EXPERIMENTOS CON ÉPOCAS Y TASA DE APRENDIZAJE")
print("=" * 65)
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

print(f"{'Experimento':<20} | {'Épocas':>6} | {'Tasa':>5} | {'Error final':>11} | {'Correctas':>9} | {'P(45%,34°)':>10}")
print("-" * 78)
for nombre, ep, tasa in experimentos:
    p, b, err = entrenar(tasa, ep)
    probs = sigmoide(X_normalizado @ p + b)
    correctas_exp = int(np.sum((probs >= 0.5).astype(int) == y.astype(int)))
    prob_45_34, _ = predecir(45, 34, p, b)
    print(f"{nombre:<20} | {ep:>6} | {tasa:>5} | {err:>11.6f} | {correctas_exp:>7}/10 | {prob_45_34:>10.4f}")

# ---------------------------------------------------------------
# 8. Prueba adicional con el umbral (sin volver a entrenar)
# ---------------------------------------------------------------
print("\n" + "=" * 65)
print("COMPARACIÓN DE UMBRALES (misma neurona base, sin reentrenar)")
print("=" * 65)
umbrales = [0.4, 0.5, 0.6]

print("\nDatos de entrenamiento:")
print(f"{'Caso':>4} | {'Humedad':>7} | {'Temp':>5} | {'Prob.':>6} | " + " | ".join(f"u={u}" for u in umbrales))
print("-" * 55)
for i, (entrada, prob) in enumerate(zip(X, probabilidades.ravel()), start=1):
    decisiones = " | ".join(f"{int(prob >= u):>5}" for u in umbrales)
    print(f"{i:>4} | {entrada[0]:>6.0f}% | {entrada[1]:>4.0f}° | {prob:>6.3f} | {decisiones}")

print("\nCondiciones nuevas:")
print(f"{'Humedad':>7} | {'Temp':>5} | {'Prob.':>6} | " + " | ".join(f"u={u}" for u in umbrales))
print("-" * 48)
for humedad, temperatura in condiciones_nuevas:
    prob, _ = predecir(humedad, temperatura, pesos, sesgo)
    decisiones = " | ".join(f"{int(prob >= u):>5}" for u in umbrales)
    print(f"{humedad:>6}% | {temperatura:>4}° | {prob:>6.3f} | {decisiones}")

print("\nEl umbral solo cambia cómo se interpreta la probabilidad;")
print("los pesos y el sesgo son los mismos en las tres columnas.")
