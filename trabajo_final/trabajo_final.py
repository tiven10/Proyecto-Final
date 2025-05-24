import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import joblib
import os

# Descargar recursos
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join([c for c in texto if c not in string.punctuation])
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if p not in stop_words]
    return ' '.join(palabras_filtradas)

# ----------------------------------
# ENTRENAMIENTO (solo se hace una vez)
# ----------------------------------
if not os.path.exists("modelo_entrenado.pkl"):
    datos = pd.read_csv('reportes_soportes.csv')
    datos['reporte'] = datos['reporte'].apply(limpiar_texto)
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(datos['reporte'])
    y = datos['categoria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    modelo = DecisionTreeClassifier(random_state=2)
    modelo.fit(X_train, y_train)

    print("Precisión:", accuracy_score(y_test, modelo.predict(X_test)))
    print(classification_report(y_test, modelo.predict(X_test)))

    joblib.dump(modelo, 'modelo_entrenado.pkl')
    joblib.dump(vectorizador, 'vectorizador_entrenado.pkl')

# Cargar modelo y vectorizador
modelo = joblib.load('modelo_entrenado.pkl')
vectorizador = joblib.load('vectorizador_entrenado.pkl')

# ----------------------------------
# LISTA ENLAZADA
# ----------------------------------
class Nodo:
    def __init__(self, reporte):
        self.reporte = reporte
        self.siguiente = None

class ListaEnlazada:
    def __init__(self):
        self.cabeza = None

    def insertar(self, reporte):
        nuevo = Nodo(reporte)
        if not self.cabeza:
            self.cabeza = nuevo
        else:
            actual = self.cabeza
            while actual.siguiente:
                actual = actual.siguiente
            actual.siguiente = nuevo
        print(f"Reporte agregado: '{reporte}'")

    def mostrar(self):
        actual = self.cabeza
        if not actual:
            print("No hay reportes pendientes.")
            return
        print("Reportes pendientes:")
        while actual:
            print(f"- {actual.reporte}")
            actual = actual.siguiente

    def clasificar(self):
        actual = self.cabeza
        if not actual:
            print("No hay reportes por clasificar.")
            return
        print("Clasificando reportes...")
        while actual:
            limpio = limpiar_texto(actual.reporte)
            X_nuevo = vectorizador.transform([limpio])
            categoria = modelo.predict(X_nuevo)[0]
            print(f" '{actual.reporte}' →  Categoría: {categoria}")
            actual = actual.siguiente
        print(" Clasificación completa.")

# ----------------------------------
# MENÚ INTERACTIVO
# ----------------------------------
def menu_interactivo():
    lista = ListaEnlazada()
    while True:
        print("\n====== MENÚ DE SOPORTE ======")
        print("1. Ingresar nuevo reporte")
        print("2. Ver reportes pendientes")
        print("3. Clasificar reportes")
        print("4. Salir")
        opcion = input("Elige una opción (1-4): ")

        if opcion == '1':
            nuevo = input("Escribe el nuevo reporte: ")
            lista.insertar(nuevo)
        elif opcion == '2':
            lista.mostrar()
        elif opcion == '3':
            lista.clasificar()
        elif opcion == '4':
            print("Saliendo del sistema. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

# Ejecutar el menú
menu_interactivo()
