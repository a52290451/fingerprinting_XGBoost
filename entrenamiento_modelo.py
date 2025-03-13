import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Directorio base donde están los sitios web
label_encoder = LabelEncoder()

data_dir = os.path.join(os.getcwd(), "webs", "combined_data_final.csv")
# Ruta del archivo combinado
labels_dir = os.path.join(os.getcwd(), "webs", "combined_data_final_labels.csv") 

# Cargar el archivo CSV
df = pd.read_csv(data_dir)
labels = pd.read_csv(labels_dir)

df["connection_id"] = label_encoder.fit_transform(df["connection_id"].values.ravel())

# Seleccionar solo las columnas relevantes que deseas utilizar para el entrenamiento
columns_to_select = [
    "frame_time_n", "avg_packet_length_n", "ip_src_count_n", "packet_count_c_n", "total_bytes_c_n", "frame_number_n", "connection_id"
]
#"packet_count_c_n", "total_bytes_c_n", "frame_number_n", "connection_id"
#"connection_id", "packet_count_c_n", "total_bytes_c_n", 
#    "avg_packet_length_c_n", "avg_interarrival_time_c_n", "interarrival_time_c_n"
#"time_diff_n", "ip_src_count_n", "avg_packet_length_n", "port_usage_n"
#"frame_time_n", "packet_length_n", "packet_count_c_n", "total_bytes_c_n", "avg_packet_length_c_n", "avg_interarrival_time_c_n", "interarrival_time_c_n"

# Seleccionar solo esas columnas del DataFrame
df_selected = df[columns_to_select]

# Usar solo un subconjunto de datos para depurar (10% de los datos)
df_small = df_selected.iloc[:int(len(df_selected) * 0.1)]
labels_small = labels.iloc[:int(len(labels) * 0.1)]

print(df_small.head(100))
# Convertir etiquetas a valores numéricos

y = label_encoder.fit_transform(labels_small.values.ravel())

print(y)


# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_small, y, test_size=0.3, random_state=42, stratify=y)

#, stratify=y
# Definir los hiperparámetros óptimos
best_params = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.7,
    'objective': 'multi:softmax',  # Ajustar según el tipo de problema
    'num_class': len(np.unique(y)),  # Número de clases en el dataset
    'eval_metric': 'mlogloss'
}

# Inicializar y entrenar el modelo
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train,verbose=2)

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_xgboost_7.pkl')

# Evaluar el modelo
predictions = model.predict(X_test)
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

# Exportar los resultados a un archivo Excel
output_excel = os.path.join(os.getcwd(), "webs", "resultados_comparacion_2.xlsx")
results.to_excel(output_excel, index=False)

print(f"Los resultados han sido exportados a {output_excel}")