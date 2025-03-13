import os
import pandas as pd
import json
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Directorio base donde están los sitios web


data_dir = os.path.join(os.getcwd(), "webs", "combined_data_final.csv")
# Ruta del archivo combinado
labels_dir = os.path.join(os.getcwd(), "webs", "combined_data_final_labels.csv") 

# Cargar el archivo CSV
df = pd.read_csv(data_dir)
labels = pd.read_csv(labels_dir)

# Seleccionar solo las columnas relevantes


# Seleccionar solo las columnas relevantes que deseas utilizar para el entrenamiento
columns_to_select = [
    "frame_time_n", "avg_packet_length_n", "ip_src_count_n", "packet_count_c_n", "total_bytes_c_n"
]

#"connection_id", "packet_count_c_n", "total_bytes_c_n", 
#    "avg_packet_length_c_n", "avg_interarrival_time_c_n", "interarrival_time_c_n"
#"time_diff_n", "ip_src_count_n", "avg_packet_length_n", "port_usage_n"

# Seleccionar solo esas columnas del DataFrame
df_selected = df[columns_to_select]

# Usar solo un subconjunto de datos para depurar (5% de los datos)
df_small = df_selected.iloc[:int(len(df_selected) * 0.06)]
labels_small = labels.iloc[:int(len(labels) * 0.06)]

print(df_small.head(100))
# Convertir etiquetas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_small).ravel()

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_small, y, test_size=0.3, random_state=42,stratify=y)

# Entrenar el modelo XGBoost
model = xgb.XGBClassifier(eval_metric="mlogloss")

# Ajustar los hiperparámetros mediante GridSearchCV
'''param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0]
}'''

param_grid = {
    'n_estimators': [100, 200],  # Menos valores
    'learning_rate': [0.05, 0.1],  # Solo dos valores
    'max_depth': [3, 5],  # Menos profundidad
    'subsample': [0.8],  # Fijo a 0.8
    'colsample_bytree': [0.8]  # Fijo a 0.8
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

#random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
#random_search.fit(X_train, y_train)

# Guardar los mejores hiperparámetros
best_params = grid_search.best_params_
best_params_path = os.path.join(os.getcwd(), "webs", "mejores_hiperparametros.json")

with open(best_params_path, "w") as f:
    json.dump(best_params, f)

print(f"Mejores hiperparámetros guardados en: {best_params_path}")

# Usar el modelo con los mejores parámetros
best_model = grid_search.best_estimator_

# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Mean Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Entrenar el modelo con los mejores parámetros
best_model.fit(X_train, y_train)

# Guardar el modelo entrenado
model_path = os.path.join(os.getcwd(), "webs", "modelo_entrenado.joblib")
joblib.dump(best_model, model_path)
print(f"Modelo guardado en: {model_path}")

# Evaluar el modelo
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

# Mostrar las métricas
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Mostrar el valor real vs el valor predicho
results = pd.DataFrame({
    "Real": y_test,
    "Predicho": predictions
})

# Guardar predicciones en un Excel
results = pd.DataFrame({"Real": y_test, "Predicho": predictions})
output_excel = os.path.join(os.getcwd(), "webs", "resultados_comparacion.xlsx")
results.to_excel(output_excel, index=False)

print(f"Los resultados han sido exportados a {output_excel}")