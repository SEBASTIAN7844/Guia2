import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_samples = 1000

data = {
    'horas_estudio_semana': np.random.uniform(1, 30, n_samples),
    'asistencia_pct': np.random.uniform(50, 100, n_samples),
    'horas_sueno': np.random.uniform(4, 10, n_samples),
    'promedio_anterior': np.random.uniform(8, 20, n_samples),
    'actividades_extra': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
    'nivel_estres': np.random.randint(1, 11, n_samples)
}

df = pd.DataFrame(data)

df['nota_final'] = (
    0.3 * df['horas_estudio_semana'] + 
    0.1 * df['asistencia_pct'] + 
    0.4 * df['promedio_anterior'] + 
    np.random.normal(0, 1.5, n_samples)
)
df['nota_final'] = df['nota_final'].clip(0, 20)

df['aprobado'] = (df['nota_final'] >= 10.5).astype(int)

for col in ['horas_estudio_semana', 'asistencia_pct', 'horas_sueno', 'promedio_anterior']:
    mask = np.random.rand(n_samples) < 0.10
    df.loc[mask, col] = np.nan

os.makedirs('data', exist_ok=True)
df.to_csv('data/student_performance_large.csv', index=False)
print("Dataset masivo 'data/student_performance_large.csv' generado con éxito con 1000 registros.")
