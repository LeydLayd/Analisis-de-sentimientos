import os
import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

def limpiar_texto(texto):
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = ' '.join([word for word in texto.split() if word.isalpha()])
    stop_words = set(stopwords.words('english'))
    words = [word for word in texto.split() if word not in stop_words and len(word) > 1]
    return ' '.join(words)

def procesar_documentos(directorio):
    documentos = []
    for nombre_archivo in os.listdir(directorio):
        if nombre_archivo.endswith('.txt'):
            ruta_archivo = os.path.join(directorio, nombre_archivo)
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                texto = archivo.read()
                texto_limpio = limpiar_texto(texto)
                documentos.append(texto_limpio)
    return documentos

def evaluar_modelo(modelo, X_test, y_test, threshold=0.5):
    """
    Evalúa el modelo y calcula métricas de rendimiento
    """
    y_pred_proba = modelo.predict(X_test).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Métricas de Evaluación ---")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Exactitud: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negativo', 'Positivo']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    print("Procesando documentos...")
    documentos_pos = procesar_documentos('pos')
    documentos_neg = procesar_documentos('neg')
    
    todos_documentos = documentos_pos + documentos_neg
    etiquetas = np.array([1]*len(documentos_pos) + [0]*len(documentos_neg))
    
    X_train, X_test, y_train, y_test = train_test_split(
        todos_documentos, etiquetas, 
        test_size=0.2, 
        random_state=42
    )
    
    print("Preparando datos para el modelo...")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    X_train_matrix = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test_matrix = tokenizer.texts_to_matrix(X_test, mode='tfidf')
    
    print("Creando y entrenando el modelo...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_matrix.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train_matrix, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    metricas = evaluar_modelo(model, X_test_matrix, y_test)

if __name__ == "__main__":
    main()