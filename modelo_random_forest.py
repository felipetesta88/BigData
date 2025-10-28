#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

print("=== MODELO PREDICTIVO STORE CHILE ===")
print("=== CON DATOS MODIFICADOS COHERENTEMENTE ===\n")

# 1. Cargar datos MODIFICADOS
print("Cargando dataset MODIFICADO...")
df = pd.read_csv('/home/jaratesta/modelo_predictivo/dataset_modificado.csv')
print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# 2. Análisis exploratorio rápido
print("\n📊 DISTRIBUCIÓN DE VARIABLES:")
print(f"Total sesiones: 0s: {(df['total_sesiones'] == 0).sum()}, 1s: {(df['total_sesiones'] == 1).sum()}, 2+s: {(df['total_sesiones'] >= 2).sum()}")
print(f"Objetivo compra 30d: 0s: {(df['objetivo_compra_30dias'] == 0).sum()}, 1s: {(df['objetivo_compra_30dias'] == 1).sum()}")
print(f"Tiempo promedio: {df['tiempo_promedio_sesion'].mean():.1f} ± {df['tiempo_promedio_sesion'].std():.1f}")

# 3. Preprocesamiento
print("\n🔄 Preprocesando datos...")
# Codificar ciudad
le_ciudad = LabelEncoder()
df['ciudad_encoded'] = le_ciudad.fit_transform(df['ciudad'])

# 4. Definir variables
X = df[['total_sesiones', 'tiempo_promedio_sesion', 'paginas_visitadas', 'ciudad_encoded']]
y = df['objetivo_compra_30dias']

print(f"Variables predictoras: {X.shape[1]}")
print(f"Variable objetivo: {len(y.unique())} clases")

# 5. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Entrenamiento: {X_train.shape[0]} registros")
print(f"Prueba: {X_test.shape[0]} registros")

# 6. Entrenar modelo
print("\n🌲 Entrenando Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 7. Predecir y evaluar
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("📈 RESULTADOS DEL MODELO")
print("="*50)
print(f"🔹 Precisión: {accuracy_score(y_test, y_pred):.2f}")
print(f"🔹 Clases en prueba: {np.unique(y_test, return_counts=True)}")

print("\n🔍 Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 8. Importancia de características
print("\n💡 IMPORTANCIA DE VARIABLES:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# 9. Guardar modelo
joblib.dump(model, '/home/jaratesta/modelo_predictivo/modelo_rf_actualizado.pkl')
print(f"\n💾 Modelo guardado en: /home/jaratesta/modelo_predictivo/modelo_rf_actualizado.pkl")

# 10. Predicciones para todos los clientes
df['probabilidad_compra'] = model.predict_proba(X)[:, 1]
df['segmento'] = pd.cut(df['probabilidad_compra'], 
                       bins=[0, 0.3, 0.7, 1], 
                       labels=['Baja', 'Media', 'Alta'])

print("\n🎯 SEGMENTACIÓN DE CLIENTES:")
segmentacion = df['segmento'].value_counts()
print(segmentacion)

# 11. Análisis de segmentos
print("\n📊 ANÁLISIS POR SEGMENTO:")
for segmento in ['Baja', 'Media', 'Alta']:
    segmento_data = df[df['segmento'] == segmento]
    print(f"\n{segmento} probabilidad ({len(segmento_data)} clientes):")
    print(f"  • Tiempo promedio: {segmento_data['tiempo_promedio_sesion'].mean():.1f}s")
    print(f"  • Sesiones promedio: {segmento_data['total_sesiones'].mean():.1f}")
    print(f"  • Compras realizadas: {segmento_data['compras_realizadas'].sum()}")

# 12. Guardar resultados
df.to_csv('/home/jaratesta/modelo_predictivo/resultados_predicciones_actualizado.csv', index=False)
print(f"\n💾 Resultados guardados en: /home/jaratesta/modelo_predictivo/resultados_predicciones_actualizado.csv")

print("\n✅ MODELO COMPLETADO EXITOSAMENTE")
