import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def cargar_y_preprocesar_datos(filepath, strategy_impute='mean', test_size_param=0.25):
    df = pd.read_csv(filepath)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    features_cols = [c for c in df.columns if c not in ['nota_final', 'aprobado']]
    X = df[features_cols]
    y_reg = df['nota_final']
    y_cls = df['aprobado']
    
    print(f"Características: {X.shape[1]} variables")
    print(f"Valores nulos en X: {X.isnull().sum().sum()}")
    
    imputer_x = SimpleImputer(strategy=strategy_impute)
    X_imputed = imputer_x.fit_transform(X)
    print(f"Valores imputados en X con estrategia: {strategy_impute}")
    
    imputer_y = SimpleImputer(strategy='mean')
    y_reg_imputed = imputer_y.fit_transform(y_reg.values.reshape(-1, 1)).ravel()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print("Características escaladas (StandardScaler)")
    
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X_scaled, y_reg_imputed, y_cls,
        test_size=test_size_param, random_state=42
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
