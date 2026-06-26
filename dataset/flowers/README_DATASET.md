# Dataset de flores

Coloca las imágenes reales dentro de cada carpeta:

- `daisy/`
- `dandelion/`
- `rose/`
- `sunflower/`
- `tulip/`

El cargador del Bloque 1 reconoce `.jpg`, `.jpeg`, `.png` y `.bmp` usando Windows Imaging Component.

Por defecto solo carga algunas imágenes por clase, según `max.images.per.class` en `config/config.txt`, para verificar rápido que el flujo funciona.
