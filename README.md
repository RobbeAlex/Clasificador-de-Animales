# 🐾 Clasificador de Animales

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un modelo de aprendizaje profundo basado en PyTorch y visión por computadora diseñado para identificar y clasificar diferentes especies de animales a partir de imágenes, todo accesible a través de una interfaz web.

## 📋 Tabla de Contenidos
- [Características](#-características)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Entrenamiento del Modelo](#-entrenamiento-del-modelo)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características
*   **Interfaz Web Interactiva:** Sube imágenes directamente desde tu navegador y obtén predicciones en tiempo real usando Streamlit.
*   **Clasificación Multiclase:** Capaz de identificar múltiples especies de animales.
*   **Motor PyTorch:** Utiliza redes neuronales convolucionales (CNN) y transfer learning (ej. ResNet, EfficientNet) para garantizar alta precisión.
*   **Rápido y Ligero:** Inferencia optimizada para ejecutarse tanto en CPU como en GPU.

## 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python 3.8+
*   **Deep Learning:** PyTorch, Torchvision
*   **Interfaz Web:** Streamlit
*   **Procesamiento de Imágenes:** Pillow (PIL), OpenCV
*   **Manejo de Datos:** NumPy, Matplotlib

## ⚙️ Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/RobbeAlex/Clasificador-de-Animales.git](https://github.com/RobbeAlex/Clasificador-de-Animales.git)
    cd Clasificador-de-Animales
    ```

2.  **Crea un entorno virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    
## ⚙️ Instrucciones de Uso
Para replicar el entrenamiento o utilizar el modelo, es muy importante seguir el orden de los siguientes pasos:

- **Paso 1: Configuración de Rutas (config.py)**
Antes de ejecutar cualquier script, debes ajustar las rutas de tu entorno local.
  * Abre el archivo config.py en tu editor de código.
  * Localiza las variables de entorno o rutas principales (por ejemplo, DATA_PATH, MODEL_SAVE_PATH, etc.).
  * Modifica las rutas para que apunten a los directorios de tu computadora donde deseas guardar el dataset descargado y el modelo entrenado.  


- **Paso 2: Descarga y Preparación de Datos (get_data.py)**
Una vez configurada la ruta, descarga y procesa el set de datos de imágenes ejecutando:
```bash
python get_data.py
```
Este script se encargará de descargar las imágenes de los animales y organizarlas en carpetas de entrenamiento y validación según lo definido en config.py.  


- **Paso 3: Construcción del Modelo (model_builder.py)**
Para inicializar y verificar la arquitectura del modelo de clasificación, ejecuta:
```bash
python model_builder.py
```
Este script construye la red neuronal. Ejecutarlo de manera independiente sirve para confirmar que la arquitectura compila correctamente y mostrar un resumen (summary) del modelo.  


- **Paso 4: Entrenamiento (train.py)**
Para comenzar a entrenar el modelo con los datos previamente descargados, ejecuta:
```bash
python train.py
```
Durante este proceso, el script leerá las imágenes, aplicará las transformaciones necesarias y entrenará la red. Al finalizar, el modelo entrenado (ej. modelo.pth o modelo.h5) se guardará en la ruta que especificaste en config.py.  

- **Paso 5: Realizar Predicciones (`predict_image.py`)**
Una vez que el modelo ha sido entrenado y guardado correctamente, puedes utilizarlo para clasificar nuevas imágenes de animales. Para hacer una predicción, ejecuta:
```bash
python predict_image.py
```
(Nota: Asegúrate de revisar el script por si necesitas pasarle la ruta de la imagen como argumento, por ejemplo: python predict_image.py --image_path ruta/a/tu/imagen.jpg. Este script cargará el modelo desde la ruta definida en config.py y mostrará qué animal detectó en la imagen).

## 🚀 Uso (Interfaz Web)
Para interactuar con el modelo a través de la aplicación web, ejecuta el siguiente comando en tu terminal:
```bash
streamlit run app.py
```

## Ejemplo de salida:
```bash
Predicción: 🐶 Perro (Confianza: 98.5%)
```
Esto abrirá automáticamente una pestaña en tu navegador web (por defecto en http://localhost:8501) donde podrás arrastrar y soltar las imágenes de los animales que deseas clasificar.

## 📁 Estructura del Proyecto
```bash
Clasificador-de-Animales/
│
├── data/                   # Datasets de entrenamiento y prueba
├── models/                 # Modelos entrenados y guardados (.pt, .pth)
├── src/                    # Código fuente principal
│   ├── config.py           # Script para establecer variables de ruta y entorno
│   ├── get_data.py         # Script para descargar y organizar el dataset
│   ├── dataset.py          # Manejo de DataLoaders y Data Augmentation
│   ├── translate.py        # Utilidad para mapear los nombres de clases a español
│   ├── model_builder.py    # Definición de la arquitectura del modelo
│   ├── engine.py           # Funciones lógicas del bucle de entrenamiento y evaluación
│   ├── train.py            # Script principal de entrenamiento
│   ├── predict_image.py    # Clasificación de imágenes nuevas mediante terminal
│   └── app.py              # Aplicación principal de Streamlit (Interfaz Web)
├── .gitignore              # Archivos y Carpetas que se debe ignorar y no rastrear
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación del proyecto
```

## 🧩 Archivos Auxiliares y de Soporte
Para mantener un código limpio, legible y modular, este proyecto cuenta con archivos que no se ejecutan de manera directa, sino que actúan como soporte para los scripts principales:

- **dataset.py:** Justifica su presencia encapsulando toda la lógica de transformación de imágenes de PyTorch. Es importado por train.py para aplicar Data Augmentation (rotaciones, normalización) y convertir las imágenes en lotes (batches) optimizados para la memoria del sistema.
- **translate.py:** El dataset original etiqueta a los animales en inglés o italiano (ej. dog, cane). Este archivo funciona como un diccionario global para estandarizar la salida al español. Es consumido por la inferencia (predict_image.py / app.py) para devolver resultados amigables (ej. "🐶 Perro").
- **engine.py:** Contiene la lógica matemática pura del entrenamiento (cálculo de pérdida, propagación hacia atrás y precisión). Esto evita que el archivo de entrenamiento principal quede sobrepoblado de código.

## 🧠 Entrenamiento del Modelo
Si deseas entrenar el modelo desde cero con tu propio dataset de animales:

Coloca tus imágenes en la carpeta data/train/ organizadas en subcarpetas por clase (ej. data/train/perros/, data/train/gatos/).

Ejecuta el script de entrenamiento:
```bash
python src/train.py --epochs 25 --batch_size 32
```

## 🤝 Contribución
¡Las contribuciones son bienvenidas! Si deseas mejorar el modelo, agregar nuevas especies o corregir errores:
* Haz un Fork del proyecto.
* Crea una nueva rama (git checkout -b feature/NuevaCaracteristica).
* Haz un Commit de tus cambios (git commit -m 'Añadir nueva característica').
* Haz Push a la rama (git push origin feature/NuevaCaracteristica).
* Abre un Pull Request.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Desarrollado por RobbeAlex
