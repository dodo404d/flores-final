# Guía para continuar después del Bloque 1

## Estado que deja este bloque

Este bloque deja listo:

- Lectura del dataset real organizado por carpetas.
- Conversión de imágenes a tensores.
- Normalización de datos.
- Estructuras base `Tensor` y `Matrix`.
- Capas densas y activación ReLU.
- Softmax con entropía cruzada.
- SGD.
- Métricas y pruebas funcionales.

## Siguiente bloque recomendado

El siguiente equipo debe implementar:

1. `Conv2D`
   - `forward()` secuencial.
   - `backward()` secuencial.
   - Validación de dimensiones.

2. `MaxPool2D`
   - `forward()`.
   - `backward()` guardando posiciones máximas.

3. `Flatten`
   - Tensor a vector.
   - Vector a Tensor para backpropagation.

4. `CNN`
   - Encadenar capas.
   - Entrenar usando imágenes reales.

5. Benchmark
   - Medir tiempos por época.
   - Guardar loss y accuracy.

## Bloque paralelo posterior

Cuando la CNN secuencial funcione, agregar OpenMP principalmente en:

- `Conv2D forward`.
- `Conv2D backward`.
- Evaluación por lote.

La comparación final debe usar la misma semilla, mismos pesos iniciales y mismo dataset.
