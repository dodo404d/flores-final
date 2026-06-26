# FlowerCNN_Bloque1

Base funcional en C++ para el proyecto **Identificación de flores mediante CNN secuencial y paralela**.

Este bloque deja preparado el proyecto para que el equipo agregue después `Conv2D`, `MaxPool2D`, `Flatten`, entrenamiento completo y paralelización con OpenMP.

## Qué contiene este bloque

- Proyecto para **Visual Studio 2022**.
- Estructura limpia por módulos: `core`, `data`, `layers`, `training`, `utils` y `tests`.
- Carga de imágenes desde carpetas reales del dataset:

```text

dataset/flowers/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

- Lectura de imágenes `.jpg`, `.jpeg`, `.png` y `.bmp` usando **Windows Imaging Component (WIC)**, sin instalar OpenCV.
- Redimensionamiento simple a `64x64`.
- Normalización de píxeles a `[0,1]`.
- Estructuras `Tensor` y `Matrix`.
- Capas base: `Dense`, `ReLU`, `SoftmaxCrossEntropy`.
- Optimizador `SGD`.
- Métrica básica de accuracy.
- Pruebas funcionales internas.
- Generación de resultados en la carpeta `results/`.

## Cómo usarlo

1. Abre `FlowerCNN_Bloque1.sln` en Visual Studio 2022.
2. Coloca tus imágenes reales dentro de:

```text

dataset/flowers/daisy/
dataset/flowers/dandelion/
dataset/flowers/rose/
dataset/flowers/sunflower/
dataset/flowers/tulip/
```

3. Compila en `Debug` o `Release`.
4. Ejecuta el proyecto.
5. Revisa la consola y la carpeta `results/`.

## Archivos generados al ejecutar

```text
results/dataset_resumen.txt
results/prueba_bloque1.csv
results/resumen_bloque1.txt
```

## Nota importante

Este bloque **no implementa todavía la CNN completa**. Su objetivo es dejar una base excelente para que el resto del equipo implemente después:

```text
Conv2D -> ReLU -> MaxPool2D -> Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Softmax
```

La paralelización con OpenMP debe integrarse en bloques posteriores, principalmente en `Conv2D forward` y `Conv2D backward`.

## Requisitos

- Windows.
- Visual Studio 2022.
- C++17.
- Windows SDK instalado con Visual Studio.

No se usa `.bat`.
