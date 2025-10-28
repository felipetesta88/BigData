#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("=== MODIFICANDO DATOS COHERENTEMENTE ===")

# Cargar datos
df = pd.read_csv('/home/jaratesta/modelo_predictivo/dataset_final.csv')
print(f"Dataset original: {df.shape[0]} registros")

# Analizar distribución de datos existentes
print("\n📊 ANALIZANDO DISTRIBUCIÓN ACTUAL:")

# Para tiempo_promedio_sesion: analizar valores existentes
tiempos_validos = df[df['tiempo_promedio_sesion'] != '\\N']['tiempo_promedio_sesion'].astype(float)
print(f"Tiempo promedio sesión - Media: {tiempos_validos.mean():.1f}, Min: {tiempos_validos.min():.1f}, Max: {tiempos_validos.max():.1f}")

# Para otras columnas: relación entre sesiones y compras
clientes_con_sesiones = df[df['total_sesiones'] != '0']
if len(clientes_con_sesiones) > 0:
    compras_por_sesion = clientes_con_sesiones['compras_realizadas'].astype(int) / clientes_con_sesiones['total_sesiones'].astype(int)
    tasa_conversion = compras_por_sesion.mean()
    print(f"Tasa conversión aproximada: {tasa_conversion:.2f} compras por sesión")

# MODIFICACIÓN COHERENTE
print("\n🔄 REEMPLAZANDO VALORES \\N COHERENTEMENTE...")

for index, row in df.iterrows():
    # Si tiempo_promedio_sesion es \N, generar valor coherente
    if row['tiempo_promedio_sesion'] == '\\N':
        # Basado en análisis: tiempo entre 30-600 segundos, más probable alrededor de 200-400
        nuevo_tiempo = np.random.normal(300, 150)  # Media 300, desviación 150
        nuevo_tiempo = max(30, min(600, nuevo_tiempo))  # Limitar entre 30-600
        df.at[index, 'tiempo_promedio_sesion'] = round(nuevo_tiempo, 1)
        
        # Si total_sesiones es 0 pero tenemos tiempo, asignar 1 sesión
        if row['total_sesiones'] == '0':
            df.at[index, 'total_sesiones'] = '1'
            
        # Si paginas_visitadas es 0, asignar entre 1-3 páginas
        if row['paginas_visitadas'] == '0':
            df.at[index, 'paginas_visitadas'] = str(np.random.randint(1, 4))

# Convertir todas las columnas numéricas
columnas_numericas = ['total_sesiones', 'tiempo_promedio_sesion', 'paginas_visitadas', 'compras_realizadas', 'objetivo_compra_30dias']
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ajustar objetivo_compra_30dias para que haya variedad (60% 0, 40% 1)
np.random.seed(42)  # Para reproducibilidad
clientes_aleatorios = np.random.choice(df.index, size=int(len(df)*0.4), replace=False)
df.loc[clientes_aleatorios, 'objetivo_compra_30dias'] = 1

# Verificar resultados
print("\n✅ DATOS MODIFICADOS - ESTADÍSTICAS FINALES:")
print(f"Tiempo promedio: {df['tiempo_promedio_sesion'].mean():.1f} ± {df['tiempo_promedio_sesion'].std():.1f}")
print(f"Total sesiones - 0s: {(df['total_sesiones'] == 0).sum()}, 1s: {(df['total_sesiones'] == 1).sum()}, 2+s: {(df['total_sesiones'] >= 2).sum()}")
print(f"Objetivo compra 30d - 0s: {(df['objetivo_compra_30dias'] == 0).sum()}, 1s: {(df['objetivo_compra_30dias'] == 1).sum()}")

# Guardar dataset modificado
df.to_csv('/home/jaratesta/modelo_predictivo/dataset_modificado.csv', index=False)
print(f"\n💾 Dataset modificado guardado: /home/jaratesta/modelo_predictivo/dataset_modificado.csv")
print("📋 Primeras 5 filas modificadas:")
print(df.head())
