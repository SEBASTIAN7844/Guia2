from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class ClasificadorAprobacion:
    def __init__(self, tipo_modelo='logistic'):
        if tipo_modelo == 'logistic':
            self.model = LogisticRegression()
        elif tipo_modelo == 'tree':
            self.model = DecisionTreeClassifier(random_state=42)
    
    def entrenar(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"Modelo de clasificación entrenado: {self.model.__class__.__name__}")
    
    def evaluar(self, X_test, y_test):
        predicciones = self.model.predict(X_test)
        acc = accuracy_score(y_test, predicciones)
        cm = confusion_matrix(y_test, predicciones)
        return acc, cm, predicciones
