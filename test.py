import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import re
from nltk.corpus import stopwords

class SentimentAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Sentiment Analyzer")
        master.geometry("500x400")

        # Load pre-trained model and tokenizer
        try:
            self.model = load_model('modelo_sentimientos.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {str(e)}")
            return

        # Create UI Components
        self.create_widgets()

    def create_widgets(self):
        # File Selection Section
        self.file_label = tk.Label(self.master, text="Select a text file to analyze:")
        self.file_label.pack(pady=10)

        self.file_path = tk.StringVar()
        self.file_entry = tk.Entry(self.master, textvariable=self.file_path, width=50)
        self.file_entry.pack(pady=5)

        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        # Analyze Button
        self.analyze_button = tk.Button(self.master, text="Analyze Sentiment", command=self.analyze_file)
        self.analyze_button.pack(pady=10)

        # Results Section
        self.result_label = tk.Label(self.master, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(self.master, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.file_path.set(filename)

    def limpiar_texto(self, texto):
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = ' '.join([word for word in texto.split() if word.isalpha()])
        stop_words = set(stopwords.words('english'))
        words = [word for word in texto.split() if word not in stop_words and len(word) > 2]
        return ' '.join(words)

    def analyze_file(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texto = f.read()
            
            texto_limpio = self.limpiar_texto(texto)
            X = self.tokenizer.texts_to_matrix([texto_limpio], mode='tfidf')
            
            prediccion = self.model.predict(X)[0][0]
            
            sentiment = "Positive" if prediccion > 0.5 else "Negative"
            confidence = prediccion * 100 if prediccion > 0.5 else (1 - prediccion) * 100
            
            self.result_label.config(text=f"Sentiment: {sentiment}", 
                                     fg="green" if sentiment == "Positive" else "red")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Could not analyze file: {str(e)}")

def main():
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
