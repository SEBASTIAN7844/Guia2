from src.data_loader import cargar_y_preprocesar_datos
from src.regression_model import RegresorNotas
from src.classification_model import ClasificadorAprobacion
import os

def main():
    print("="*70)
    print(" PIPELINE MODULAR DE APRENDIZAJE SUPERVISADO - DATASET MASIVO")
    print("="*70)
    
    # Verificar si existe el dataset masivo
    if os.path.exists('data/student_performance_large.csv'):
        filepath = 'data/student_performance_large.csv'
        print("\n[INFO] Usando dataset masivo: student_performance_large.csv")
    else:
        filepath = 'data/dataset_supervisado.csv'
        print("\n[INFO] Usando dataset base: dataset_supervisado.csv")
    
    # ============ EXPERIMENTO 1: COMPARACIÓN DE IMPUTACIÓN ============
    print("\n" + "="*70)
    print(" TAREA 1: COMPARACIÓN DE ESTRATEGIAS DE IMPUTACIÓN")
    print("="*70)
    
    estrategias = ['mean', 'median']
    
    for estrategia in estrategias:
        print(f"\n--- Estrategia: {estrategia.upper()} ---")
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = \
            cargar_y_preprocesar_datos(filepath, strategy_impute=estrategia, test_size_param=0.20)
        
        # Regresión
        regresor = RegresorNotas()
        regresor.entrenar(X_train, y_reg_train)
        mse, rmse, r2, _ = regresor.evaluar(X_test, y_reg_test)
        print(f"Regresión - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Clasificación
        clasificador = ClasificadorAprobacion(tipo_modelo='logistic')
        clasificador.entrenar(X_train, y_cls_train)
        acc, cm, _ = clasificador.evaluar(X_test, y_cls_test)
        print(f"Clasificación - Accuracy: {acc*100:.2f}%")
    
    # ============ EXPERIMENTO 2: COMPARACIÓN DE TEST SIZE ============
    print("\n" + "="*70)
    print(" TAREA 2: REGRESIÓN - COMPARACIÓN DE TAMAÑO DE PRUEBA")
    print("="*70)
    
    test_sizes = [0.20, 0.50]
    print("\n{:<12} {:<12} {:<12} {:<12}".format("Test Size", "MSE", "RMSE", "R²"))
    print("-" * 48)
    
    for test_size in test_sizes:
        # CORREGIDO: Solo desempaquetamos 6 valores, no 8
        X_train, X_test, y_reg_train, y_reg_test, _, _ = \
            cargar_y_preprocesar_datos(filepath, strategy_impute='mean', test_size_param=test_size)
        
        regresor = RegresorNotas()
        regresor.entrenar(X_train, y_reg_train)
        mse, rmse, r2, _ = regresor.evaluar(X_test, y_reg_test)
        print("{:<12} {:<12.4f} {:<12.4f} {:<12.4f}".format(test_size, mse, rmse, r2))
    
    # ============ EXPERIMENTO 3: COMPARACIÓN DE CLASIFICADORES ============
    print("\n" + "="*70)
    print(" TAREA 3: CLASIFICACIÓN - COMPARACIÓN DE ALGORITMOS")
    print("="*70)
    
    # CORREGIDO: Solo desempaquetamos 6 valores
    X_train, X_test, _, _, y_cls_train, y_cls_test = \
        cargar_y_preprocesar_datos(filepath, strategy_impute='mean', test_size_param=0.20)
    
    modelos = [
        ('Logistic Regression', 'logistic'),
        ('Decision Tree', 'tree')
    ]
    
    print("\n{:<25} {:<15} {:<20}".format("Modelo", "Accuracy", "Matriz de Confusión"))
    print("-" * 70)
    
    for nombre, tipo in modelos:
        clasificador = ClasificadorAprobacion(tipo_modelo=tipo)
        clasificador.entrenar(X_train, y_cls_train)
        acc, cm, _ = clasificador.evaluar(X_test, y_cls_test)
        print("{:<25} {:<15.2f}% {:<20}".format(nombre, acc*100, str(cm.tolist())))
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()
