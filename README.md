# Analisis de sentimientos con redes neuronales en Python
Este proyecto implementa un modelo de redes neuronales para el analisis de sentimientos en reseñas de peliculas, ya que puede detectar si una reseña es positiva o negativa, esto utilizando Python y TensorFlow/Keras.


## Caracteristicas

 - Limpieza y preprocesamiento de texto.
 - Tokenizacion y vectorizacion con TF-IDF.
 - Clasificacion binaria de sentimientos (Positivo/Negativo).
 - Evaluacion del modelo con métricas de desempeño.

Este proyecto esta desarrollado tanto en Google Colaboratory en linea como una version para computadora de escritorio.

## Estructura del proyecto

    /
    ├── neg/                  # Archivos de entrenamiento positivos
    ├── pos/                  # Archivos de entrenamiento negativos
    ├── prb_neg/              # Archivos de prueba
    ├── prb_pos/              # Archivos de prueba
    ├── Analizador.ipynb      # Vesion de Colaboratory
    ├── main.py               # Script principal
    ├── test.py               # Script para probar el modelo
    ├── README.md             # Este archivo


## Requisitos Colaboratory
Para ejecutarlo este proyecto en Google Colaboratory([Google Colab](https://colab.research.google.com/?hl=es)) solo necesita de una cuenta de Google.
## Uso
Sigue estos pasos

 1. Descarga el archivo `Analizador.ipynb` desde este repositorio.
 2.  Abre Google Colab y sube el archivo como cuaderno.
 3. Ejecuta las siguientes celdas en orden:
	 - **Importacion de librerias**
	 - **Clonacion del repositorio**
	 - **Entrenamiento del modelo**

### Probar el modelo
Una vez que el entrenamiento haya finalizado, ejecuta la seccion de **Pruebas**, y sigue estos pasos:

 1. Presiona el boton **Cargar archivo.txt**.
 2. Selecciona un archivo desde tu ordenador en formato .**.txt** para analizar.
 3. Haz clic en **Analizar sentimiento**.

El sistema determinara si la reseña es **positiva o negativa**.

## Requisitos PC

Asegurate de tener instaladas las siguientes librerias antes de ejecutar el codigo:

    pip install numpy nltk tensorflow scikit-learn     
> **Nota:** Asegúrate de ejecutar `nltk.download('stopwords')` antes de entrenar el modelo, para evitar errores relacionados con la ausencia de las palabras vacías en inglés.


## Uso
### Entrenar el modelo

Ejecuta el siguiente comando para procesar el modelo y evaluar su rendimiento:

**Windows:**

    python main.py
**Mac/Linux:**

    python3 main.py

### Probar el modelo
Para evaluar los textos, usa el script `test_red.py` en el cual se te mostrara una ventana para poder elegir el texto a analizar desde el explorador de archivos:

**Windows:**

    python test_model.py
**Mac/Linux:**

    python3 test_model.py

El sistema determinara si la reseña es **positiva o negativa**

## Evaluacion del modelo

El script `main.py` muestra métricas como:

 - Precisión
 - Recall
 - Exactitud
 - F1-Score
 - Matriz de confusión

## Autor
[Diego Robles](https://github.com/LeydLayd)

