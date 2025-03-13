import os
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar un dataset de ejemplo

data_dir = os.path.join(os.getcwd(), "webs", "combined_data_final.csv")
# Ruta del archivo combinado
labels_dir = os.path.join(os.getcwd(), "webs", "combined_data_final_labels.csv") 

# Cargar el archivo CSV
df = pd.read_csv(data_dir)
labels = pd.read_csv(labels_dir)


# Usar solo un subconjunto de datos para depurar (5% de los datos)
df_small = df.iloc[:int(len(df) * 0.03)]
labels_small = labels.iloc[:int(len(labels) * 0.03)]

# Convertir etiquetas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_small)
y=y.ravel()

df_small["ip.src"] = label_encoder.fit_transform(df_small["ip.src"]).ravel()

print("IP SRC")
print (df["ip.src"])
df_small["connection_id"] = label_encoder.fit_transform(df_small["connection_id"]).ravel()

print("Conexion")
print (df_small["connection_id"])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_small, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo XGBoost
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# Obtener la importancia de las características
feature_importance = model.feature_importances_

# Convertirlo en un DataFrame para mejor visualización
importance_df = pd.DataFrame({'Feature': df.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Mostrar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de Características en XGBoost")
plt.gca().invert_yaxis()  # Invertir el eje para mostrar la más importante arriba
plt.show()