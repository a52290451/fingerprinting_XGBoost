import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Inicializar el escalador MinMax
scaler = MinMaxScaler()

# Directorio base donde están los sitios web

data_dir = os.path.join(os.getcwd(), "webs", "combined_data.csv") # Ruta de datos
output_csv = os.path.join(os.getcwd(), "webs", "combined_data_final.csv")  # Ruta del archivo de salida
output_csv_labels = os.path.join(os.getcwd(), "webs", "combined_data_final_labels.csv") #Ruta de salida de labels finales
output_excel = os.path.join(os.getcwd(), "webs", "combined_data_final_sample.xlsx")  # Ruta del archivo de Excel con el 10% de los datos

# Cargar inicial de los datos
df = pd.read_csv(data_dir)

# Muestra inicial
print(df.head(100))

#Eliminación de registros con valores faltantes
df = df[df['ip.proto'].notna() & (df['ip.proto'] != '')]
df = df[df['tcp.srcport'].notna() & (df['tcp.srcport'] != '')]
#df = df[df['udp.srcport'].notna() & (df['udp.srcport'] != '')]
print("PARTE 2---------------------")

#Muestra del DF resultante
print(df.head(100))

#Rellenar valores nulos con 0
df.fillna(0,inplace=True)

# Separar etiquetas (labels) y datos (df sin la columna 'site')
if "site" in df.columns:
    labels = df["site"]
    df = df.drop(columns=["site"])
else:
    labels = None

# Mostrar información de los datos
print("Primeras filas del DataFrame:")
print(df.head(100))

print("\nPrimeras etiquetas:")
print(labels.head(100) if labels is not None else "No se encontraron etiquetas.")

# Asegurar que ip.src es una cadena y manejar valores nulos
df["ip.src"] = df["ip.src"].astype(str).fillna("")

# Eliminar la zona horaria antes de convertir fram.time
df["frame_time"] = df["frame.time"].str.replace(r" \w{3,4}$", "", regex=True)
 
# Convertir 'frame.time' a datetime
df["frame_time"] = pd.to_datetime(df["frame.time"], errors='coerce')
 
#Convertir el datetime a un valor numérico (por ejemplo, segundos desde la época Unix)
df["frame_time"] = df["frame_time"].astype(int) / 10**9  # Dividir por 10^9 para convertir nanosegundos a segundos

# Normalizar la columna 'frame_time'
df["frame_time_n"] = scaler.fit_transform(df[['frame_time']])

# Ver las primeras filas después de la normalización
print(df[['frame_time', 'frame_time_n']].head())

# Tamaño de los paquetes normalizado
df["packet_length"] = df["frame.len"]
df["packet_length_n"] = scaler.fit_transform(df[["frame.len"]])

# Protocolo normalizado
df["protocol"] = scaler.fit_transform(df[["ip.proto"]])

# Normalización del numero de frame
df["frame_number_n"] = scaler.fit_transform(df[["frame.number"]])

# Características adicionales

# Diferencia de tiempo normalizada con respecto al registro anterior.
df["time_diff"] = df["frame_time"].diff().abs().fillna(0)
df["time_diff_n"] = scaler.fit_transform(df[['time_diff']])

# Cantidad de paquetes enviados por IP - Normalizado
df["ip_src_count"] = df.groupby("ip.src")["frame.number"].transform("count")
df["ip_src_count_n"] = scaler.fit_transform(df[["ip_src_count"]])

# Promedio del tamaño de paquetes por IP - Normalizado
df["avg_packet_length"] = df.groupby("ip.src")["packet_length"].transform("mean")
df["avg_packet_length_n"] = scaler.fit_transform(df[["avg_packet_length"]])

# Cantidad de veces que un puerto ha sido usado - Normalizado
df["port_usage"] = df.groupby("tcp.srcport")["frame.number"].transform("count")
df["port_usage_n"] = scaler.fit_transform(df[["port_usage"]])

#Generar ID unico por conexión
df["connection_id"] = df["ip.src"].astype(str) + "_" + df["ip.proto"].astype(str) + "_" + df["tcp.srcport"].astype(str)

#Nuevas caracteristicas basadas en el trafico con el nuevo ID

# cantidad de paquetes por conexión
df["packet_count_c"] = df.groupby("connection_id")["frame.number"].transform("count")
df["packet_count_c_n"] = scaler.fit_transform(df[["packet_count_c"]])

# cantidad de bytes en total, por conexión
df["total_bytes_c"] = df.groupby("connection_id")["frame.len"].transform("sum")
df["total_bytes_c_n"] = scaler.fit_transform(df[["total_bytes_c"]])

# Promedio de bytes por conexión
df["avg_packet_length_c"] = df.groupby("connection_id")["frame.len"].transform("mean")
df["avg_packet_length_c_n"] = scaler.fit_transform(df[["avg_packet_length_c"]])

# Promedio de tiempo de captura entre paquetes de la misma conexión.
df["avg_interarrival_time_c"] = df.groupby("connection_id")["frame_time"].transform(lambda x: x.diff().abs().mean()).fillna(0)
df["avg_interarrival_time_c_n"] = scaler.fit_transform(df[["avg_interarrival_time_c"]])
df["interarrival_time_c"] = df.groupby("connection_id")["frame_time"].transform(lambda x: x.diff().abs()).fillna(0)
df["interarrival_time_c_n"] = scaler.fit_transform(df[["interarrival_time_c"]])



df = df[["ip.src","frame_time_n","packet_length_n","protocol","frame_number_n","time_diff_n","ip_src_count_n","avg_packet_length_n","port_usage_n","connection_id","packet_count_c_n","total_bytes_c_n","avg_packet_length_c_n","avg_interarrival_time_c_n","interarrival_time_c_n"]]

print("\nMuestra final")
print(df.head(10000))

# Tomar el 5% de los datos
df_sample = df.sample(frac=0.05, random_state=42)  # frac=0.05 selecciona el 5%

# Guardar el 5% de los datos en un archivo Excel
df_sample.to_excel(output_excel, index=False)

df.to_csv(output_csv, index=False)

labels.to_csv(output_csv_labels, index=False)