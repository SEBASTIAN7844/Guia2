import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class RegresorNotas:
    def __init__(self):
        self.model = LinearRegression()
    
    def entrenar(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"Modelo de regresión entrenado")
    
    def evaluar(self, X_test, y_test):
        predicciones = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predicciones)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predicciones)
        return mse, rmse, r2, predicciones
